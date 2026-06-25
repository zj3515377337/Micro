"""Prompt 组装与上下文预算控制。

这个模块负责决定：每一轮到底把多少 prefix、memory、相关笔记、历史
以及当前用户请求送进模型。

从单阶段压缩升级为 5 阶段自适应压缩（OPENDEV §3.2.2）：
- Stage 1  (≥70%)：警告模式，仅记录监控，不改变 prompt
- Stage 2  (≥80%)：观察遮蔽——旧工具输出替换为引用指针
- Stage 2.5 (≥85%)：快速修剪——删除保护窗口外的旧输出
- Stage 3  (≥90%)：激进遮蔽——仅保留最近 3 轮 + 压缩预算
- Stage 4  (≥99%)：全量 LLM 摘要（需 Compact 模型，可选）
"""

from __future__ import annotations

import json
from dataclasses import dataclass


DEFAULT_TOTAL_BUDGET = 12000
DEFAULT_SECTION_BUDGETS = {
    "prefix": 4800,
    "memory": 1600,
    "relevant_memory": 1200,
    "history": 5200,
}
DEFAULT_SECTION_FLOORS = {
    "prefix": 1200,
    "memory": 400,
    "relevant_memory": 300,
    "history": 1500,
}
# 当 prompt 超预算时，会优先压缩这些 section。
DEFAULT_REDUCTION_ORDER = ("relevant_memory", "history", "memory", "prefix")
SECTION_ORDER = ("prefix", "memory", "relevant_memory", "history", "current_request")
CURRENT_REQUEST_SECTION = "current_request"
RELEVANT_MEMORY_LIMIT = 3

# ── 自适应压缩阈值（可配置）──────────────────────────────────────────
STAGE_1_WARNING = 0.70       # ≥70% → 日志记录
STAGE_2_OBSERVATION_MASK = 0.80  # ≥80% → 旧工具输出变引用指针
STAGE_2_5_FAST_PRUNE = 0.85      # ≥85% → 删除旧输出 + 快速修剪
STAGE_3_AGGRESSIVE_MASK = 0.90   # ≥90% → 仅保留最近 3 轮
STAGE_4_FULL_COMPACTION = 0.99   # ≥99% → LLM 摘要（最后手段）

# 各阶段的历史保护窗口大小
PROTECTION_WINDOW_NORMAL = 6    # Stage 0/1/2：最近 6 轮完整保留
PROTECTION_WINDOW_TIGHT = 3     # Stage 2.5/3：最近 3 轮完整保留
PROTECTION_WINDOW_MINIMAL = 1   # Stage 4：仅保留最后一轮


def _tail_clip(text, limit):
    text = str(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


@dataclass
class SectionRender:
    raw: str
    budget: int
    rendered: str
    details: dict | None = None

    @property
    def raw_chars(self):
        return len(self.raw)

    @property
    def rendered_chars(self):
        return len(self.rendered)


class ContextManager:
    def __init__(
        self,
        agent,
        total_budget=DEFAULT_TOTAL_BUDGET,
        section_budgets=None,
        section_floors=None,
        reduction_order=None,
    ):
        self.agent = agent
        self.total_budget = int(total_budget)
        self.section_budgets = dict(DEFAULT_SECTION_BUDGETS)
        if section_budgets:
            self.section_budgets.update({str(key): int(value) for key, value in section_budgets.items()})
        self._section_floor_overrides = {str(key): int(value) for key, value in (section_floors or {}).items()}
        self.section_floors = self._compute_section_floors()
        self.reduction_order = tuple(reduction_order or DEFAULT_REDUCTION_ORDER)

    def build(self, user_message):
        """按预算组装一轮完整 prompt，含自适应压缩。

        流程：
        1. 组装 section 文本 + 检索相关记忆
        2. 估算当前使用率
        3. 按使用率分阶段压缩（而非一刀切）
        4. 返回 prompt + metadata
        """
        user_message = str(user_message)
        self.section_floors = self._compute_section_floors()
        memory_enabled = True
        relevant_memory_enabled = True
        context_reduction_enabled = True
        if hasattr(self.agent, "feature_enabled"):
            memory_enabled = self.agent.feature_enabled("memory")
            relevant_memory_enabled = self.agent.feature_enabled("relevant_memory")
            context_reduction_enabled = self.agent.feature_enabled("context_reduction")
        section_texts = {
            "prefix": str(getattr(self.agent, "prefix", "")),
            "memory": "Memory:\n- disabled" if not memory_enabled else str(self.agent.memory_text()),
            "history": "",
            CURRENT_REQUEST_SECTION: f"Current user request:\n{user_message}",
        }
        checkpoint_text = ""
        if hasattr(self.agent, "render_checkpoint_text"):
            checkpoint_text = str(self.agent.render_checkpoint_text() or "").strip()
        if checkpoint_text:
            section_texts["prefix"] = section_texts["prefix"] + "\n\n" + checkpoint_text
        # Project Knowledge（ACE Playbook 跨会话知识）
        playbook = getattr(self.agent, "memory", None)
        if playbook and hasattr(playbook, "playbook_text"):
            pk_text = playbook.playbook_text()
            if pk_text:
                section_texts["prefix"] = section_texts["prefix"] + "\n\n" + pk_text

        # 活跃计划注入（Plan Mode 的输出）
        active_plan = getattr(self.agent, "active_plan", "")
        if active_plan:
            section_texts["prefix"] = section_texts["prefix"] + "\n\nCurrent Plan:\n" + active_plan
        selected_notes = []
        if memory_enabled and relevant_memory_enabled and hasattr(self.agent, "memory") and hasattr(self.agent.memory, "retrieval_candidates"):
            selected_notes = self.agent.memory.retrieval_candidates(user_message, limit=RELEVANT_MEMORY_LIMIT)

        if not context_reduction_enabled:
            rendered = self._render_sections_without_reduction(section_texts, selected_notes=selected_notes)
            prompt = self._assemble_prompt(rendered)
            metadata = self._metadata(
                prompt=prompt,
                rendered=rendered,
                budgets={section: render.budget for section, render in rendered.items() if section != CURRENT_REQUEST_SECTION},
                reduction_log=[],
                selected_notes=selected_notes,
                user_message=user_message,
                section_texts=section_texts,
            )
            return prompt, metadata

        budgets = dict(self.section_budgets)
        rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
        prompt = self._assemble_prompt(rendered)
        reduction_log = []

        # ── 自适应压缩：按使用率分阶段 ──
        usage_ratio = self._estimate_usage_ratio(prompt)

        if usage_ratio >= STAGE_4_FULL_COMPACTION:
            # Stage 4：全量 LLM 摘要（回退到激进遮蔽 + 硬裁剪）
            reduction_log.append({"section": "history", "stage": 4, "strategy": "full_compaction", "usage_ratio": usage_ratio})
            rendered, budgets, prompt, stage_log = self._apply_aggressive_reduction(
                rendered, section_texts, budgets, selected_notes, protection_window=PROTECTION_WINDOW_MINIMAL
            )
            reduction_log.extend(stage_log)

        elif usage_ratio >= STAGE_3_AGGRESSIVE_MASK:
            # Stage 3：激进遮蔽——仅保留最近 3 轮 + 深度压缩所有 section 预算
            reduction_log.append({"section": "history", "stage": 3, "strategy": "aggressive_mask", "usage_ratio": usage_ratio})
            rendered, budgets, prompt, stage_log = self._apply_aggressive_reduction(
                rendered, section_texts, budgets, selected_notes, protection_window=PROTECTION_WINDOW_TIGHT
            )
            reduction_log.extend(stage_log)

        elif usage_ratio >= STAGE_2_5_FAST_PRUNE:
            # Stage 2.5：快速修剪——观察遮蔽 + 缩减保护窗口 + 预算压缩
            reduction_log.append({"section": "history", "stage": 2.5, "strategy": "fast_prune", "usage_ratio": usage_ratio})
            rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes,
                                              observation_mask=True, protection_window=PROTECTION_WINDOW_TIGHT)
            prompt = self._assemble_prompt(rendered)
            # 结合预算压缩
            rendered, budgets, prompt, stage_log = self._apply_aggressive_reduction(
                rendered, section_texts, budgets, selected_notes, protection_window=PROTECTION_WINDOW_TIGHT
            )
            reduction_log.extend(stage_log)

        elif usage_ratio >= STAGE_2_OBSERVATION_MASK:
            # Stage 2：观察遮蔽——旧工具输出替换为引用指针，但保留最近 6 轮
            reduction_log.append({"section": "history", "stage": 2, "strategy": "observation_mask", "usage_ratio": usage_ratio})
            rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes,
                                              observation_mask=True, protection_window=PROTECTION_WINDOW_NORMAL)
            prompt = self._assemble_prompt(rendered)

        elif usage_ratio >= STAGE_1_WARNING:
            # Stage 1：仅记录监控数据，不改变 prompt
            reduction_log.append({"section": "system", "stage": 1, "strategy": "warning", "usage_ratio": usage_ratio})

        # Stage 0（<70%）：什么也不做，正常返回

        # ── 兜底：如果 prompt 仍然超预算，执行传统压缩 ──
        if len(prompt) > self.total_budget:
            rendered, budgets, prompt, fallback_log = self._apply_fallback_reduction(
                rendered, section_texts, budgets, selected_notes, prompt
            )
            reduction_log.extend(fallback_log)

        metadata = self._metadata(
            prompt=prompt,
            rendered=rendered,
            budgets=budgets,
            reduction_log=reduction_log,
            selected_notes=selected_notes,
            user_message=user_message,
            section_texts=section_texts,
        )
        return prompt, metadata

    # ── 自适应压缩辅助方法 ───────────────────────────────────────────

    def _estimate_usage_ratio(self, prompt):
        """估算 prompt 占预算的比例。"""
        if self.total_budget <= 0:
            return 0.0
        # 优先用 API 返回的实际 token 数
        meta = getattr(self.agent, "last_completion_metadata", None) or {}
        input_tokens = int(meta.get("input_tokens", 0) or 0)
        if input_tokens > 0:
            model_limit = getattr(self.agent.model_client, "context_window", None)
            if model_limit and model_limit > 0:
                return input_tokens / model_limit
        # 回退到字符数估算
        return len(prompt) / self.total_budget

    def _apply_aggressive_reduction(self, rendered, section_texts, budgets, selected_notes,
                                     protection_window=PROTECTION_WINDOW_TIGHT):
        """Stage 2.5/3/4 共享：重渲染（含遮蔽 + 收紧保护窗口）+ 预算压缩。"""
        rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes,
                                          observation_mask=True, protection_window=protection_window)
        prompt = self._assemble_prompt(rendered)

        # 在已遮蔽的基础上继续压缩 section 预算
        tight_floors = {
            "prefix": max(400, self.section_floors.get("prefix", 1200) // 2),
            "memory": max(150, self.section_floors.get("memory", 400) // 2),
            "relevant_memory": max(80, self.section_floors.get("relevant_memory", 300) // 3),
            "history": max(400, self.section_floors.get("history", 1500) // 3),
        }

        while len(prompt) > self.total_budget:
            overflow = len(prompt) - self.total_budget
            reduced = False
            for section in self.reduction_order:
                floor = tight_floors.get(section, 20)
                current_budget = int(budgets.get(section, 0))
                if current_budget <= floor:
                    continue
                new_budget = max(floor, current_budget - overflow)
                if new_budget >= current_budget:
                    continue
                budgets[section] = new_budget
                rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes,
                                                  observation_mask=True, protection_window=protection_window)
                prompt = self._assemble_prompt(rendered)
                reduced = True
                break
            if not reduced:
                break

        return rendered, budgets, prompt

    def _apply_aggressive_reduction(self, rendered, section_texts, budgets, selected_notes,
                                     protection_window=PROTECTION_WINDOW_TIGHT):
        """Stage 2.5/3/4 共享：重渲染（含遮蔽 + 收紧保护窗口）+ 预算压缩。"""
        rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes,
                                          observation_mask=True, protection_window=protection_window)
        prompt = self._assemble_prompt(rendered)

        # 在已遮蔽的基础上继续压缩 section 预算
        tight_floors = {
            "prefix": max(400, self.section_floors.get("prefix", 1200) // 2),
            "memory": max(150, self.section_floors.get("memory", 400) // 2),
            "relevant_memory": max(80, self.section_floors.get("relevant_memory", 300) // 3),
            "history": max(400, self.section_floors.get("history", 1500) // 3),
        }

        reduction_log = []
        while len(prompt) > self.total_budget:
            overflow = len(prompt) - self.total_budget
            reduced = False
            for section in self.reduction_order:
                floor = tight_floors.get(section, 20)
                current_budget = int(budgets.get(section, 0))
                if current_budget <= floor:
                    continue
                new_budget = max(floor, current_budget - overflow)
                if new_budget >= current_budget:
                    continue
                reduction_log.append({
                    "section": section,
                    "before_chars": current_budget,
                    "after_chars": new_budget,
                    "overflow_chars": overflow,
                })
                budgets[section] = new_budget
                rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes,
                                                  observation_mask=True, protection_window=protection_window)
                prompt = self._assemble_prompt(rendered)
                reduced = True
                break
            if not reduced:
                break

        return rendered, budgets, prompt, reduction_log

    def _apply_fallback_reduction(self, rendered, section_texts, budgets, selected_notes, prompt):
        """兜底：传统渐进压缩。"""
        fallback_log = []
        while len(prompt) > self.total_budget:
            overflow = len(prompt) - self.total_budget
            reduced = False
            for section in self.reduction_order:
                floor = int(self.section_floors.get(section, 0))
                current_budget = int(budgets.get(section, 0))
                if current_budget <= floor:
                    continue
                new_budget = max(floor, current_budget - overflow)
                if new_budget >= current_budget:
                    continue
                fallback_log.append({
                    "section": section,
                    "before_chars": current_budget,
                    "after_chars": new_budget,
                    "overflow_chars": overflow,
                })
                budgets[section] = new_budget
                rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
                prompt = self._assemble_prompt(rendered)
                reduced = True
                break
            if not reduced:
                break
        return rendered, budgets, prompt, fallback_log

    # ── 渲染方法 ─────────────────────────────────────────────────────

    def _render_sections_without_reduction(self, section_texts, selected_notes=None):
        selected_notes = selected_notes or []
        relevant_lines = ["Relevant memory:"]
        if selected_notes:
            relevant_lines.extend(f"- {note['text']}" for note in selected_notes)
        else:
            relevant_lines.append("- none")
        relevant_raw = "\n".join(relevant_lines)
        history = list(getattr(self.agent, "session", {}).get("history", []))
        history_raw = self._raw_history_text(history)
        return {
            "prefix": SectionRender(raw=section_texts["prefix"], budget=len(section_texts["prefix"]), rendered=section_texts["prefix"], details={}),
            "memory": SectionRender(raw=section_texts["memory"], budget=len(section_texts["memory"]), rendered=section_texts["memory"], details={}),
            "relevant_memory": SectionRender(
                raw=relevant_raw, budget=len(relevant_raw), rendered=relevant_raw,
                details={
                    "selected_notes": [note["text"] for note in selected_notes],
                    "rendered_notes": [note["text"] for note in selected_notes],
                    "selected_count": len(selected_notes), "rendered_count": len(selected_notes), "note_budget": 0,
                },
            ),
            "history": SectionRender(raw=history_raw, budget=len(history_raw), rendered=history_raw, details={"rendered_entries": []}),
            CURRENT_REQUEST_SECTION: SectionRender(raw=section_texts[CURRENT_REQUEST_SECTION], budget=0, rendered=section_texts[CURRENT_REQUEST_SECTION], details={}),
        }

    def _compute_section_floors(self):
        floors = {
            section: max(20, int(budget) // 4)
            for section, budget in self.section_budgets.items()
        }
        floors.update(self._section_floor_overrides)
        return floors

    def _render_sections(self, section_texts, budgets, selected_notes=None,
                          observation_mask=False, protection_window=PROTECTION_WINDOW_NORMAL):
        rendered = {}
        for section in SECTION_ORDER:
            budget = budgets.get(section)
            if section == CURRENT_REQUEST_SECTION:
                raw = section_texts[section]
                rendered[section] = SectionRender(raw=raw, budget=0, rendered=raw, details={})
            elif section == "relevant_memory":
                rendered[section] = self._render_relevant_memory(selected_notes or [], int(budget or 0))
            elif section == "history":
                rendered[section] = self._render_history_section(
                    int(budget or 0), observation_mask=observation_mask, protection_window=protection_window
                )
            else:
                raw = section_texts[section]
                rendered_text = _tail_clip(raw, int(budget)) if budget is not None else raw
                rendered[section] = SectionRender(raw=raw, budget=int(budget) if budget is not None else 0, rendered=rendered_text, details={})
        return rendered

    def _render_relevant_memory(self, selected_notes, budget):
        header = "Relevant memory:"
        note_texts = [str(note.get("text", "")) for note in selected_notes if str(note.get("text", "")).strip()]
        raw_lines = [header] + [f"- {text}" for text in note_texts]
        raw = "\n".join(raw_lines) if note_texts else "\n".join([header, "- none"])
        if not note_texts:
            rendered = raw
            return SectionRender(raw=raw, budget=budget, rendered=rendered, details={
                "selected_notes": [], "rendered_notes": [], "selected_count": 0, "rendered_count": 0, "note_budget": 0,
            })

        per_note_budget = self._per_note_budget(budget, len(note_texts), header)
        rendered_notes = []
        while True:
            rendered_notes = [_tail_clip(text, per_note_budget) for text in note_texts]
            rendered = "\n".join([header] + [f"- {text}" for text in rendered_notes])
            if len(rendered) <= budget or per_note_budget <= 1:
                break
            per_note_budget -= 1

        if len(rendered) > budget and budget > 0:
            rendered = _tail_clip(raw, budget)
            rendered_notes = [rendered]

        return SectionRender(raw=raw, budget=budget, rendered=rendered, details={
            "selected_notes": note_texts, "rendered_notes": rendered_notes,
            "selected_count": len(note_texts), "rendered_count": len(rendered_notes), "note_budget": per_note_budget,
        })

    def _per_note_budget(self, budget, note_count, header):
        if note_count <= 0:
            return 0
        overhead = len(header) + 3 * note_count
        usable = max(0, budget - overhead)
        return max(1, usable // note_count)

    # ── 历史渲染（含观察遮蔽）────────────────────────────────────────

    def _render_history_section(self, budget, observation_mask=False, protection_window=PROTECTION_WINDOW_NORMAL):
        history = list(getattr(self.agent, "session", {}).get("history", []))
        raw = self._raw_history_text(history)
        if not history:
            rendered = "Transcript:\n- empty"
            return SectionRender(raw=raw, budget=budget, rendered=rendered, details={
                "rendered_entries": [], "older_entries_count": 0, "collapsed_duplicate_reads": 0,
                "reused_file_summary_count": 0, "summarized_tool_count": 0, "observation_mask": observation_mask,
            })

        recent_start = max(0, len(history) - protection_window)
        history_entries, history_details = self._compressed_history_entries(
            history, recent_start, observation_mask=observation_mask
        )
        rendered_entries = []
        for entry in reversed(history_entries):
            recent = bool(entry.get("recent", False))
            candidate_lines = list(entry.get("lines", []))
            candidate_entries = candidate_lines + rendered_entries
            candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
            if len(candidate_rendered) <= budget:
                rendered_entries = candidate_entries
                continue
            if recent:
                available = budget - len("Transcript:")
                if rendered_entries:
                    available -= sum(len(line) + 1 for line in rendered_entries)
                available = max(20, available - 1)
                candidate_lines = [_tail_clip(line, available) for line in candidate_lines]
                candidate_entries = candidate_lines + rendered_entries
                candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
                if len(candidate_rendered) <= budget:
                    rendered_entries = candidate_entries
            else:
                smaller_lines = [_tail_clip(line, 20) for line in candidate_lines]
                smaller_entries = smaller_lines + rendered_entries
                smaller_rendered = "\n".join(["Transcript:", *smaller_entries])
                if len(smaller_rendered) <= budget:
                    rendered_entries = smaller_entries
        rendered = "\n".join(["Transcript:", *rendered_entries])

        if len(rendered) > budget and budget > 0:
            rendered = _tail_clip(raw, budget)

        return SectionRender(raw=raw, budget=budget, rendered=rendered, details={
            "recent_window": protection_window, "recent_start": recent_start,
            "rendered_entries": rendered_entries, "observation_mask": observation_mask,
            **history_details,
        })

    def _compressed_history_entries(self, history, recent_start, observation_mask=False):
        entries = []
        seen_older_reads = set()
        details = {
            "older_entries_count": 0,
            "collapsed_duplicate_reads": 0,
            "reused_file_summary_count": 0,
            "summarized_tool_count": 0,
        }

        for index, item in enumerate(history):
            recent = index >= recent_start
            if recent:
                line_limit = 900
                entries.append({"recent": True, "lines": self._render_history_item(item, line_limit)})
                continue

            if item["role"] == "tool" and item["name"] == "read_file":
                path = str(item["args"].get("path", "")).strip()
                if path in seen_older_reads:
                    details["collapsed_duplicate_reads"] += 1
                    continue
                seen_older_reads.add(path)
                summary = self._reusable_file_summary(path)
                if summary:
                    entries.append({"recent": False, "lines": [f"{path} -> {summary}"]})
                    details["older_entries_count"] += 1
                    details["reused_file_summary_count"] += 1
                    continue

            if item["role"] == "tool":
                if observation_mask:
                    summary_line = self._mask_tool_output(item)
                else:
                    summary_line = self._summarize_old_tool_item(item)
                entries.append({"recent": False, "lines": [summary_line]})
                details["older_entries_count"] += 1
                details["summarized_tool_count"] += 1
                continue

            entries.append({"recent": False, "lines": self._render_history_item(item, 60)})

        return entries, details

    def _mask_tool_output(self, item):
        """Stage 2 核心：将旧工具输出替换为结构化引用指针。

        与 _summarize_old_tool_item 的区别：
        - 后者只是截断原始内容
        - 本方法提取结构化元信息（行数、匹配数、退出码、关键符号等）
        """
        name = item["name"]
        args = item.get("args", {})
        content = str(item.get("content", ""))

        if name == "read_file":
            path = str(args.get("path", "")).strip()
            lines = content.splitlines()
            body_lines = [l for l in lines if not l.startswith("# ")]  # 去掉 # path 头
            line_count = len(body_lines)
            # 提取关键符号（函数/类定义）—— 需去掉行号前缀
            symbols = []
            for line in body_lines[:50]:
                # read_file 输出格式："   1: import os" → 提取 "import os"
                no_prefix = line.split(": ", 1)[-1] if ": " in line else line
                stripped = no_prefix.strip()
                if stripped.startswith("def ") or stripped.startswith("class "):
                    sym = stripped.split("(")[0].replace("def ", "").replace("class ", "").strip()
                    if sym and sym not in symbols:
                        symbols.append(sym)
            sym_text = f", defines: {', '.join(symbols[:6])}" if symbols else ""
            return f"[tool:read_file] {path} → ({line_count} lines{sym_text})"

        if name == "search":
            pattern = str(args.get("pattern", "")).strip()
            match_lines = [l for l in content.splitlines() if l.strip() and not l.startswith("# ")]
            match_count = len(match_lines)
            # 统计涉及的文件数
            files = set()
            for line in match_lines:
                if ":" in line:
                    files.add(line.split(":")[0])
            file_text = f" in {len(files)} files" if len(files) > 1 else ""
            return f"[tool:search] pattern='{pattern}' → ({match_count} matches{file_text})"

        if name == "run_shell":
            command = str(args.get("command", "")).strip()[:60] or "shell"
            exit_code = "?"
            for line in content.splitlines():
                if line.strip().startswith("exit_code:"):
                    exit_code = line.split(":", 1)[1].strip()
            passed = failed = 0
            for line in content.splitlines():
                if "passed" in line.lower() or "failed" in line.lower():
                    import re
                    pm = re.search(r"(\d+)\s+passed", line)
                    fm = re.search(r"(\d+)\s+failed", line)
                    if pm: passed = int(pm.group(1))
                    if fm: failed = int(fm.group(1))
            if passed or failed:
                return f"[tool:run_shell] {command} → (exit:{exit_code}, {passed}P/{failed}F)"
            return f"[tool:run_shell] {command} → (exit:{exit_code}, output masked)"

        # 其他工具：简洁一行
        return f"[tool:{name}] (output masked)"

    def _reusable_file_summary(self, path):
        memory = getattr(self.agent, "memory", None)
        if memory is None or not hasattr(memory, "to_dict"):
            return ""
        snapshot = memory.to_dict()
        summary = snapshot.get("file_summaries", {}).get(str(path), {})
        if not summary:
            return ""
        return str(summary.get("summary", "")).strip()

    def _summarize_old_tool_item(self, item):
        """Stage 0 使用的旧工具摘要（截断而非遮蔽）。"""
        if item["name"] == "run_shell":
            command = str(item["args"].get("command", "")).strip() or "shell"
            lines = [line.strip() for line in str(item.get("content", "")).splitlines() if line.strip()]
            summary = " | ".join(lines[:3]) if lines else "(empty)"
            return f"{command} -> {summary}"
        return self._render_history_item(item, 60)[0]

    def _raw_history_text(self, history):
        if not history:
            return "Transcript:\n- empty"
        lines = []
        for item in history:
            if item["role"] == "tool":
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(str(item["content"]))
            else:
                lines.append(f"[{item['role']}] {item['content']}")
        return "\n".join(["Transcript:", *lines])

    def _render_history_item(self, item, line_limit):
        if item["role"] == "tool":
            prefix = f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}"
            content = _tail_clip(item["content"], max(20, line_limit))
            return [prefix, content]
        return [f"[{item['role']}] {_tail_clip(item['content'], line_limit)}"]

    def _assemble_prompt(self, rendered):
        return "\n\n".join(
            [
                rendered["prefix"].rendered,
                rendered["memory"].rendered,
                rendered["relevant_memory"].rendered,
                rendered["history"].rendered,
                rendered[CURRENT_REQUEST_SECTION].rendered,
            ]
        ).strip()

    def _metadata(self, prompt, rendered, budgets, reduction_log, selected_notes, user_message, section_texts):
        section_metadata = {}
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {
                "raw_chars": rendered[section].raw_chars,
                "budget_chars": int(budgets.get(section, 0)),
                "rendered_chars": rendered[section].rendered_chars,
            }
        section_metadata[CURRENT_REQUEST_SECTION] = {
            "raw_chars": len(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_chars": None,
            "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
        }
        return {
            "prompt_chars": len(prompt),
            "prompt_budget_chars": self.total_budget,
            "prompt_over_budget": len(prompt) > self.total_budget,
            "section_order": list(SECTION_ORDER),
            "section_budgets": {
                section: (None if section == CURRENT_REQUEST_SECTION else int(budgets.get(section, 0)))
                for section in SECTION_ORDER
            },
            "sections": section_metadata,
            "budget_reductions": reduction_log,
            "reduction_order": list(self.reduction_order),
            "relevant_memory": {
                "limit": RELEVANT_MEMORY_LIMIT,
                "selected_count": len(selected_notes),
                "selected_notes": [note["text"] for note in selected_notes],
                "selected_sources": [str(note.get("source", "")).strip() for note in selected_notes],
                "selected_kinds": [str(note.get("kind", "episodic")).strip() or "episodic" for note in selected_notes],
                "selected_durable_count": sum(
                    1 for note in selected_notes if (str(note.get("kind", "episodic")).strip() or "episodic") == "durable"
                ),
                "raw_chars": rendered["relevant_memory"].raw_chars,
                "rendered_chars": rendered["relevant_memory"].rendered_chars,
                "rendered_notes": list(rendered["relevant_memory"].details.get("rendered_notes", [])),
                "rendered_count": int(rendered["relevant_memory"].details.get("rendered_count", 0)),
            },
            "history": {
                "raw_chars": rendered["history"].raw_chars,
                "rendered_chars": rendered["history"].rendered_chars,
                "older_entries_count": int(rendered["history"].details.get("older_entries_count", 0)),
                "collapsed_duplicate_reads": int(rendered["history"].details.get("collapsed_duplicate_reads", 0)),
                "reused_file_summary_count": int(rendered["history"].details.get("reused_file_summary_count", 0)),
                "summarized_tool_count": int(rendered["history"].details.get("summarized_tool_count", 0)),
                "observation_mask": bool(rendered["history"].details.get("observation_mask", False)),
            },
            "current_request": {
                "text": user_message,
                "raw_chars": len(user_message),
                "rendered_chars": len(user_message),
                "section_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
            },
        }
