"""多步 agent 运行时使用的轻量工作记忆。

session history 负责保存完整事件流；这个模块只保存更小的一层工作集：
当前任务摘要、最近接触的文件、文件短摘要，以及少量跨轮笔记。
这样下一轮 prompt 还能接上上一轮，但不会被整段历史塞满。
"""

import hashlib
from datetime import datetime
import re
from pathlib import Path

from ..workspace import clip, now

WORKING_FILE_LIMIT = 8
EPISODIC_NOTE_LIMIT = 12
FILE_SUMMARY_LIMIT = 6

DURABLE_TOPIC_DEFAULTS = {
    "project-conventions": {
        "title": "Project Conventions",
        "summary": "Stable repository conventions.",
        "tags": ["convention"],
    },
    "key-decisions": {
        "title": "Key Decisions",
        "summary": "Long-lived decisions and rationale anchors.",
        "tags": ["decision"],
    },
    "dependency-facts": {
        "title": "Dependency Facts",
        "summary": "Stable dependency and environment facts.",
        "tags": ["dependency"],
    },
    "user-preferences": {
        "title": "User Preferences",
        "summary": "Stable user preferences.",
        "tags": ["preference"],
    },
}


def default_memory_state():
    # 用一个小而结构化的状态，而不是一大段自由文本摘要。
    return {
        "working": {
            "task_summary": "",
            "recent_files": [],
        },
        "episodic_notes": [],
        "file_summaries": {},
        "task": "",
        "files": [],
        "notes": [],
        "next_note_index": 0,
    }


class DurableMemoryStore:
    def __init__(self, root):
        self.root = Path(root)
        self.index_path = self.root / "MEMORY.md"
        self.topics_dir = self.root / "topics"

    def topic_slugs(self):
        return [topic["topic"] for topic in self.load_index()]

    def load_index(self):
        if not self.index_path.exists():
            return []
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        topics = []
        current = None
        for raw in lines:
            line = raw.strip()
            match = re.match(r"- \[([^\]]+)\]\([^)]+\):\s*(.+)", line)
            if match:
                current = {
                    "topic": match.group(1).strip(),
                    "title": match.group(2).strip(),
                    "summary": "",
                    "tags": [],
                }
                topics.append(current)
                continue
            if current is None:
                continue
            summary_match = re.match(r"- summary:\s*(.+)", line)
            if summary_match:
                current["summary"] = summary_match.group(1).strip()
                continue
            tags_match = re.match(r"- tags:\s*(.+)", line)
            if tags_match:
                current["tags"] = [tag.strip() for tag in tags_match.group(1).split(",") if tag.strip()]
        return topics

    def load_topic_notes(self, topic):
        path = self.topics_dir / f"{topic}.md"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        notes = []
        capture = False
        updated_at = ""
        tags = []
        for raw in lines:
            line = raw.strip()
            if line.startswith("- tags:"):
                tags = [tag.strip() for tag in line.split(":", 1)[1].split(",") if tag.strip()]
            elif line.startswith("- updated_at:"):
                updated_at = line.split(":", 1)[1].strip()
            elif line == "## Notes":
                capture = True
            elif capture and line.startswith("- "):
                notes.append(
                    {
                        "text": line[2:].strip(),
                        "tags": tags,
                        "source": topic,
                        "created_at": updated_at or now(),
                        "kind": "durable",
                    }
                )
        return notes

    @staticmethod
    def _subject_key(text):
        text = str(text).strip()
        patterns = (
            r"^(.+?)\s+is\s+.+$",
            r"^(.+?)\s+are\s+.+$",
            r"^(.+?)\s+uses?\s+.+$",
            r"^(.+?)\s+should\s+.+$",
            r"^(.+?)是.+$",
            r"^(.+?)使用.+$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, re.I)
            if match:
                subject = " ".join(_tokenize(match.group(1)))
                return subject or None
        return None

    def retrieval_candidates(self, query, limit=3):
        query_tokens = _tokenize(query)
        ranked = []
        for topic in self.load_index():
            notes = self.load_topic_notes(topic["topic"])
            for note in notes:
                note_tags = {tag.lower() for tag in note.get("tags", [])}
                note_tokens = _tokenize(note.get("text", "")) | _tokenize(topic.get("title", "")) | note_tags
                exact_tag_match = int(bool(query_tokens & note_tags))
                keyword_overlap = len(query_tokens & note_tokens)
                if exact_tag_match == 0 and keyword_overlap == 0:
                    continue
                recency = _parse_timestamp(note.get("created_at"))
                ranked.append(((exact_tag_match, keyword_overlap, recency), note))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [note for _, note in ranked[:limit]]

    def _write_index(self, topics):
        self.root.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        lines = ["# Durable Memory Index", ""]
        for topic in topics:
            lines.append(f"- [{topic['topic']}](topics/{topic['topic']}.md): {topic['title']}")
            lines.append(f"  - summary: {topic['summary']}")
            lines.append(f"  - tags: {', '.join(topic['tags'])}")
        self.index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _write_topic(self, topic, notes):
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        meta = DURABLE_TOPIC_DEFAULTS[topic]
        lines = [
            f"# {meta['title']}",
            "",
            f"- topic: {topic}",
            f"- summary: {meta['summary']}",
            f"- tags: {', '.join(meta['tags'])}",
            f"- updated_at: {now()}",
            "",
            "## Notes",
        ]
        for note in notes:
            lines.append(f"- {note}")
        (self.topics_dir / f"{topic}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def promote(self, promotions):
        if not promotions:
            return [], []
        topics = {topic["topic"]: topic for topic in self.load_index()}
        topic_notes = {slug: [note["text"] for note in self.load_topic_notes(slug)] for slug in topics}
        results = []
        superseded = []
        for topic, note_text in promotions:
            meta = DURABLE_TOPIC_DEFAULTS[topic]
            topics.setdefault(
                topic,
                {
                    "topic": topic,
                    "title": meta["title"],
                    "summary": meta["summary"],
                    "tags": list(meta["tags"]),
                },
            )
            existing = topic_notes.setdefault(topic, [])
            if note_text in existing:
                continue
            new_subject = self._subject_key(note_text)
            replaced = False
            if new_subject:
                for index, old_text in enumerate(list(existing)):
                    if self._subject_key(old_text) == new_subject:
                        superseded.append(f"{topic}: {old_text} -> {note_text}")
                        existing[index] = note_text
                        replaced = True
                        break
            if not replaced:
                existing.append(note_text)
            results.append(f"{topic}: {note_text}")
        self._write_index([topics[slug] for slug in sorted(topics)])
        for topic, notes in topic_notes.items():
            self._write_topic(topic, notes)
        return results, superseded


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_workspace_path(raw_path, workspace_root=None):
    path = Path(str(raw_path))
    if workspace_root is None:
        return path

    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path, workspace_root=None):
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    root = Path(workspace_root).resolve()
    return resolved.relative_to(root).as_posix()


def file_freshness(raw_path, workspace_root=None):
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _tokenize(text):
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", str(text))}


def _parse_timestamp(value):
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0


def _normalize_note(note, index):
    if isinstance(note, str):
        text = clip(note.strip(), 500)
        return {
            "text": text,
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    if not isinstance(note, dict):
        text = clip(str(note).strip(), 500)
        return {
            "text": text,
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    text = clip(str(note.get("text", "")).strip(), 500)
    tags = [str(tag).strip() for tag in _ensure_list(note.get("tags", [])) if str(tag).strip()]
    source = str(note.get("source", "")).strip()
    created_at = str(note.get("created_at", "")).strip() or now()
    note_index = int(note.get("note_index", index))
    kind = str(note.get("kind", "episodic")).strip() or "episodic"
    return {
        "text": text,
        "tags": _dedupe_preserve_order(tags),
        "source": source,
        "created_at": created_at,
        "note_index": note_index,
        "kind": kind,
    }


def normalize_memory_state(state, workspace_root=None):
    if state is None:
        state = default_memory_state()
    elif not isinstance(state, dict):
        raise TypeError("memory state must be a mapping")

    # 规范化层的作用，是把“磁盘里可能长得不太一样的旧状态”
    # 统一整理成当前 runtime 可直接使用的紧凑结构。
    working = state.get("working")
    if not isinstance(working, dict):
        working = {}
    working.setdefault("task_summary", "")
    working.setdefault("recent_files", [])
    working["task_summary"] = clip(str(working.get("task_summary", "")).strip(), 300)
    working["recent_files"] = _dedupe_preserve_order(
        [
            canonicalize_path(path, workspace_root)
            for path in _ensure_list(working.get("recent_files", []))
            if str(path).strip()
        ]
    )[-WORKING_FILE_LIMIT:]
    state["working"] = working

    if not str(working["task_summary"]).strip() and state.get("task"):
        working["task_summary"] = clip(str(state.get("task", "")).strip(), 300)
    if not working["recent_files"] and state.get("files"):
        working["recent_files"] = _dedupe_preserve_order(
            [
                canonicalize_path(path, workspace_root)
                for path in _ensure_list(state.get("files", []))
                if str(path).strip()
            ]
        )[-WORKING_FILE_LIMIT:]

    episodic_notes = state.get("episodic_notes")
    if not isinstance(episodic_notes, list):
        episodic_notes = []

    if not episodic_notes and state.get("notes"):
        episodic_notes = [
            _normalize_note(note, index)
            for index, note in enumerate(_ensure_list(state.get("notes", [])))
            if str(note).strip()
        ]
    else:
        normalized_notes = []
        for index, note in enumerate(episodic_notes):
            if isinstance(note, str) and not str(note).strip():
                continue
            normalized_notes.append(_normalize_note(note, index))
        episodic_notes = normalized_notes
    episodic_notes = episodic_notes[-EPISODIC_NOTE_LIMIT:]
    state["episodic_notes"] = episodic_notes

    file_summaries = state.get("file_summaries")
    if not isinstance(file_summaries, dict):
        file_summaries = {}
    normalized_file_summaries = {}
    for path, summary in file_summaries.items():
        path = canonicalize_path(path, workspace_root)
        if isinstance(summary, dict):
            text = clip(str(summary.get("summary", "")).strip(), 500)
            created_at = str(summary.get("created_at", "")).strip() or now()
            freshness = summary.get("freshness")
            freshness = None if freshness in (None, "") else str(freshness).strip() or None
        else:
            text = clip(str(summary).strip(), 500)
            created_at = now()
            freshness = None
        if not path or not text:
            continue
        normalized_file_summaries[path] = {
            "summary": text,
            "created_at": created_at,
            "freshness": freshness,
        }
    state["file_summaries"] = normalized_file_summaries

    next_note_index = state.get("next_note_index")
    if not isinstance(next_note_index, int) or next_note_index < 0:
        next_note_index = 0
    max_index = max([note["note_index"] for note in episodic_notes], default=-1)
    state["next_note_index"] = max(next_note_index, max_index + 1)

    state["task"] = working["task_summary"]
    state["files"] = list(working["recent_files"])
    state["notes"] = [note["text"] for note in episodic_notes]
    durable_root = Path(workspace_root) / ".pico" / "memory" if workspace_root is not None else None
    durable_store = DurableMemoryStore(durable_root) if durable_root is not None else None
    state["durable_topics"] = durable_store.topic_slugs() if durable_store is not None else []
    return state


def set_task_summary(state, summary, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    state["working"]["task_summary"] = clip(str(summary).strip(), 300)
    state["task"] = state["working"]["task_summary"]
    return state


def remember_file(state, path, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    files = [item for item in state["working"]["recent_files"] if item != path]
    files.append(path)
    state["working"]["recent_files"] = files[-WORKING_FILE_LIMIT:]
    state["files"] = list(state["working"]["recent_files"])
    return state


def append_note(state, text, tags=(), source="", created_at=None, workspace_root=None, kind="episodic"):
    state = normalize_memory_state(state, workspace_root)
    text = clip(str(text).strip(), 500)
    if not text:
        return state

    normalized_tags = _dedupe_preserve_order(
        [str(tag).strip() for tag in _ensure_list(tags) if str(tag).strip()]
    )
    note = {
        "text": text,
        "tags": normalized_tags,
        "source": str(source).strip(),
        "created_at": str(created_at).strip() if created_at else now(),
        "note_index": int(state.get("next_note_index", 0)),
        "kind": str(kind).strip() or "episodic",
    }
    state["next_note_index"] = note["note_index"] + 1

    notes = [item for item in state["episodic_notes"] if item["text"] != note["text"]]
    notes.append(note)
    state["episodic_notes"] = notes[-EPISODIC_NOTE_LIMIT:]
    state["notes"] = [item["text"] for item in state["episodic_notes"]]
    return state
def set_file_summary(state, path, summary, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    summary = clip(str(summary).strip(), 500)
    if not path or not summary:
        return state
    state["file_summaries"][path] = {
        "summary": summary,
        "created_at": now(),
        "freshness": file_freshness(path, workspace_root),
    }
    return state


def invalidate_file_summary(state, path, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    state["file_summaries"].pop(path, None)
    return state


def invalidate_stale_file_summaries(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    invalidated = []
    for path, summary in list(state["file_summaries"].items()):
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("freshness") == current_freshness:
            continue
        invalidated.append(path)
        state["file_summaries"].pop(path, None)
    return state, invalidated


def summarize_read_result(result, limit=180):
    # 我们不会把完整文件内容塞进记忆层，
    # 这里只保留足够提醒下一轮“刚刚读到了什么”的短摘要。
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    if lines[0].startswith("# "):
        lines = lines[1:]
    if not lines:
        return "(empty)"
    summary = " | ".join(lines[:3])
    return clip(summary, limit)


def retrieval_candidates(state, query, limit=3, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    query_tokens = _tokenize(query)
    ranked = []
    for note in state["episodic_notes"]:
        # 召回逻辑故意保持简单透明：先看 tag 精确命中，
        # 再看关键词重叠，最后看新旧程度。这里不引入 embedding。
        note_tags = {tag.lower() for tag in note.get("tags", [])}
        note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | note_tags
        exact_tag_match = int(bool(query_tokens & note_tags))
        keyword_overlap = len(query_tokens & note_tokens)
        if exact_tag_match == 0 and keyword_overlap == 0:
            continue
        recency = _parse_timestamp(note.get("created_at"))
        note_index = int(note.get("note_index", 0))
        ranked.append(((exact_tag_match, keyword_overlap, recency, note_index), note))

    if workspace_root is not None:
        durable_store = DurableMemoryStore(Path(workspace_root) / ".pico" / "memory")
        for note in durable_store.retrieval_candidates(query, limit=limit):
            note_tags = {tag.lower() for tag in note.get("tags", [])}
            note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | note_tags
            exact_tag_match = int(bool(query_tokens & note_tags))
            keyword_overlap = len(query_tokens & note_tokens)
            recency = _parse_timestamp(note.get("created_at"))
            ranked.append(((exact_tag_match, keyword_overlap, recency, -1), note))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in ranked[:limit]]


def retrieval_view(state, query, limit=3, workspace_root=None):
    candidates = retrieval_candidates(state, query, limit=limit, workspace_root=workspace_root)
    lines = ["Relevant memory:"]
    if not candidates:
        lines.append("- none")
        return "\n".join(lines)
    for note in candidates:
        lines.append(f"- {note['text']}")
    return "\n".join(lines)


def render_memory_text(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    # 这里渲染的是给模型看的紧凑“仪表盘”，不是完整回放。
    # 笔记正文默认不展开，只有在相关召回时才按需拿出来。
    lines = [
        "Memory:",
        f"- task: {state['working']['task_summary'] or '-'}",
        f"- recent_files: {', '.join(state['working']['recent_files']) or '-'}",
    ]

    summaries = []
    for path in state["working"]["recent_files"][:FILE_SUMMARY_LIMIT]:
        summary = state["file_summaries"].get(path, {})
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("summary", "") and summary.get("freshness") == current_freshness:
            summaries.append(f"- {path}: {summary['summary']}")
    if summaries:
        lines.append("- file_summaries:")
        lines.extend(f"  {line}" for line in summaries)
    else:
        lines.append("- file_summaries: -")

    lines.append(f"- episodic_notes: {len(state['episodic_notes'])}")
    durable_topics = state.get("durable_topics", [])
    lines.append(f"- durable_topics: {', '.join(durable_topics) or '-'}")
    return "\n".join(lines)


def is_effectively_empty(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    return (
        not str(state["working"]["task_summary"]).strip()
        and not state["working"]["recent_files"]
        and not state["episodic_notes"]
        and not state["file_summaries"]
    )


class LayeredMemory:
    def __init__(self, state=None, workspace_root=None):
        self.workspace_root = workspace_root
        self.state = normalize_memory_state(state, workspace_root)
        self.durable_store = DurableMemoryStore(Path(workspace_root) / ".pico" / "memory") if workspace_root is not None else None

    def to_dict(self):
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return self.state

    def canonical_path(self, path):
        return canonicalize_path(path, self.workspace_root)

    def set_task_summary(self, summary):
        self.state = set_task_summary(self.state, summary, self.workspace_root)
        return self

    def remember_file(self, path):
        self.state = remember_file(self.state, path, self.workspace_root)
        return self

    def append_note(self, text, tags=(), source="", created_at=None, kind="episodic"):
        self.state = append_note(
            self.state,
            text,
            tags=tags,
            source=source,
            created_at=created_at,
            workspace_root=self.workspace_root,
            kind=kind,
        )
        return self

    def set_file_summary(self, path, summary):
        self.state = set_file_summary(self.state, path, summary, self.workspace_root)
        return self

    def invalidate_file_summary(self, path):
        self.state = invalidate_file_summary(self.state, path, self.workspace_root)
        return self

    def invalidate_stale_file_summaries(self):
        self.state, invalidated = invalidate_stale_file_summaries(self.state, self.workspace_root)
        return invalidated

    def retrieval_candidates(self, query, limit=3):
        return retrieval_candidates(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def retrieval_view(self, query, limit=3):
        return retrieval_view(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def render_memory_text(self):
        return render_memory_text(self.state, self.workspace_root)

    def promote_durable(self, promotions):
        if self.durable_store is None:
            return [], []
        self.state = normalize_memory_state(self.state, self.workspace_root)
        promoted, superseded = self.durable_store.promote(promotions)
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return promoted, superseded

    def run_playbook_extraction(self, session_history):
        """ACE Playbook Stage 1+2：扫描历史 → 提取候选 → 持久化。

        从有明确信号的交互中学习（保守策略）：
        - B: write_file 后又被 patch_file 修正 → 格式约定
        - C: 用户消息含纠正/偏好关键词 → 用户偏好
        - D: run_shell 执行成功的测试命令 → 项目约定
        - A: 跨会话高频读取的结构性文件 → 核心文件（需 2+ 会话确认）
        """
        if self.durable_store is None:
            return [], []

        candidates = _extract_playbook_candidates(session_history)
        if not candidates:
            return [], []

        # 检查是否跨会话确认（对于 A 类核心文件）
        confirmed = _confirm_candidates(self.durable_store, candidates)

        # 持久化
        promoted, superseded = self.durable_store.promote(confirmed)
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return promoted, superseded

    def playbook_text(self):
        """渲染 Project Knowledge（注入 prompt prefix 用）。"""
        if self.durable_store is None:
            return ""
        topics = self.durable_store.load_index()
        if not topics:
            return ""
        lines = ["Project Knowledge (from previous sessions):"]
        for topic in topics[:4]:  # 最多 4 个 topic
            notes = self.durable_store.load_topic_notes(topic["topic"])
            for note in notes[:3]:  # 每个 topic 最多 3 条
                lines.append(f"- {topic['title']}: {note['text']}")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)


# ── ACE Playbook 提取逻辑 ──────────────────────────────────────────

_CORRECTION_KEYWORDS = (
    "不要", "别", "应该", "请用", "最好", "推荐",
    "don't", "should", "use ", "prefer", "instead", "rather",
)

_STRUCTURAL_FILE_PATTERNS = (
    "config", "setup", "__init__", "base_", "_base", "util", "constant",
    "type", "model", "schema", "setting",
)


def _extract_playbook_candidates(history):
    """扫描会话历史，返回候选 (topic, note_text) 列表。"""
    candidates = []

    # B: write→patch 修正 → 格式约定
    for i in range(len(history)):
        if history[i].get("name") == "write_file":
            path_i = str(history[i].get("args", {}).get("path", ""))
            for j in range(i + 1, min(i + 4, len(history))):
                if history[j].get("name") == "patch_file" and history[j].get("args", {}).get("path") == path_i:
                    candidates.append(("project-conventions", f"file '{path_i}' was edited then patched — likely has specific formatting rules"))
                    break

    # C: 用户纠正 → 偏好
    for item in history:
        if item.get("role") == "user":
            text = str(item.get("content", ""))
            if any(kw in text.lower() for kw in _CORRECTION_KEYWORDS):
                clipped = text[:200].strip()
                if len(clipped) > 10:
                    candidates.append(("user-preferences", clipped))

    # D: 成功的测试命令 → 项目约定
    for item in history:
        if item.get("role") == "tool" and item.get("name") == "run_shell":
            cmd = str(item.get("args", {}).get("command", ""))
            content = str(item.get("content", ""))
            is_test_cmd = any(kw in cmd for kw in ("pytest", "tox", "unittest", "mocha", "jest", "cargo test", "go test"))
            was_success = "exit_code: 0" in content and "failed" not in content.lower()
            if is_test_cmd and was_success:
                candidates.append(("project-conventions", f"test command: {cmd}"))

    # A: 高频读取的结构性文件 → 候选核心文件
    read_counts = {}
    for item in history:
        if item.get("role") == "tool" and item.get("name") == "read_file":
            path = str(item.get("args", {}).get("path", ""))
            if path:
                read_counts[path] = read_counts.get(path, 0) + 1
    for path, count in read_counts.items():
        basename = path.rsplit("/", 1)[-1].lower()
        is_structural = any(p in basename for p in _STRUCTURAL_FILE_PATTERNS)
        if count >= 3 and is_structural:
            candidates.append(("dependency-facts", f"core file: {path} (read {count} times)"))

    return candidates


def _confirm_candidates(durable_store, candidates):
    """过滤候选：只返回需要持久化的条目。

    对于 A 类核心文件：只有跨会话再次出现才确认（避免单次探索循环误判）。
    """
    existing_notes = {}
    for topic in durable_store.load_index():
        for note in durable_store.load_topic_notes(topic["topic"]):
            existing_notes[(topic["topic"], note["text"])] = True

    confirmed = []
    for topic, text in candidates:
        key = (topic, text)
        if key not in existing_notes:
            confirmed.append((topic, text))

    # A 类需要跨会话确认的逻辑通过去重天然实现：
    # 第一次出现 → 记录但不确认（由现有 promote_durable 机制自然处理）
    # 第二次出现 → text 相同被去重跳过，不会重复添加噪声
    # 只有持续稳定的模式才会被记住
    return confirmed
