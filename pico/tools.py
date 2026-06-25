"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import shutil
import subprocess
import textwrap
from functools import partial

from .workspace import IGNORED_PATH_NAMES

BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "description": "List files in the workspace.",
    },
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "risky": False,
        "description": "Read a UTF-8 file by line range.",
    },
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
        "description": "Search the workspace with rg or a simple fallback.",
    },
    "run_shell": {
        "schema": {"command": "str", "timeout": "int=20"},
        "risky": True,
        "description": "Run a shell command in the repo root.",
    },
    "write_file": {
        "schema": {"path": "str", "content": "str"},
        "risky": True,
        "description": "Write a text file.",
    },
    "patch_file": {
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "risky": True,
        "description": "Replace a text block in a file (fuzzy matching, 6 stages).",
    },
    "file_info": {
        "schema": {"path": "str"},
        "risky": False,
        "description": "Get file size, line count, and modification time.",
    },
    "glob": {
        "schema": {"pattern": "str='**/*'"},
        "risky": False,
        "description": "Find files matching a glob pattern (e.g. src/**/*.py).",
    },
    "grep_count": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
        "description": "Count matches per file without returning content.",
    },
    "git_diff": {
        "schema": {"staged": "bool=False"},
        "risky": False,
        "description": "Show working tree changes as a unified diff.",
    },
    "git_log": {
        "schema": {"n": "int=5", "path": "str='.'"},
        "risky": False,
        "description": "Show recent git commit history.",
    },
}

DELEGATE_TOOL_SPEC = {
    "schema": {"task": "str", "max_steps": "int=3"},
    "risky": False,
    "description": "Ask a bounded read-only child agent to investigate.",
}


def legal_tool_names():
    return set(BASE_TOOL_SPECS) | {"delegate"}

TOOL_EXAMPLES = {
    "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
    "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
    "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
    "file_info": '<tool>{"name":"file_info","args":{"path":"src/main.py"}}</tool>',
    "glob": '<tool>{"name":"glob","args":{"pattern":"src/**/*.py"}}</tool>',
    "grep_count": '<tool>{"name":"grep_count","args":{"pattern":"TODO","path":"."}}</tool>',
    "git_diff": '<tool>{"name":"git_diff","args":{}}</tool>',
    "git_log": '<tool>{"name":"git_log","args":{"n":5}}</tool>',
}


def build_tool_registry(context):
    # 工具不是动态发现的，而是显式注册的。
    # 这样模型看到的是一个有边界、可审计的动作集合。
    tools = {
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], context)}
        for name, spec in BASE_TOOL_SPECS.items()
    }
    # 子 agent 是刻意做成受限能力的：一旦深度耗尽，
    # 就连 delegate 这个工具都不再暴露给模型。
    if context.depth < context.max_depth:
        tools["delegate"] = {**DELEGATE_TOOL_SPEC, "run": partial(tool_delegate, context)}
    return tools


def tool_example(name):
    return TOOL_EXAMPLES.get(name, "")


def validate_tool(context, name, args):
    args = args or {}

    if name == "list_files":
        path = context.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return

    if name == "read_file":
        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        return

    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        context.path(args.get("path", "."))
        return

    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        return

    if name == "write_file":
        path = context.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
        return

    if name == "patch_file":
        # patch_file 使用 6 阶段渐进式模糊匹配（OPENDEV Lesson 4）。
        # 这里只做基础校验：路径存在、old_text 非空、new_text 存在。
        # 匹配逻辑在执行阶段完成。
        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        return

    if name == "file_info":
        context.path(args["path"])
        return

    if name == "glob":
        pattern = str(args.get("pattern", "**/*")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        return

    if name == "grep_count":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        context.path(args.get("path", "."))
        return

    if name == "git_diff":
        return

    if name == "git_log":
        path = context.path(args.get("path", "."))
        n = int(args.get("n", 5))
        if n < 1 or n > 50:
            raise ValueError("n must be in [1, 50]")
        return

    if name == "delegate":
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        if context.depth >= context.max_depth:
            raise ValueError("delegate depth exceeded")
        return


def tool_list_files(context, args):
    path = context.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(context.root)}")
    return "\n".join(lines) or "(empty)"


def tool_read_file(context, args):
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    return f"# {path.relative_to(context.root)}\n{body}"


def tool_search(context, args):
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = context.path(args.get("path", "."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=context.root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"

    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(context.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(context.root)}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def tool_run_shell(context, args):
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    result = subprocess.run(
        command,
        cwd=context.root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        # 这里传入的是过滤后的环境变量，而不是直接继承整个父 shell 环境，
        # 目的是减少敏感信息被意外带进命令执行环境的风险。
        env=context.shell_env(),
    )
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {result.stderr.strip() or "(empty)"}
        """
    ).strip()


def tool_write_file(context, args):
    path = context.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(context.root)} ({len(content)} chars)"


def tool_patch_file(context, args):
    """6 阶段渐进式模糊匹配替换。

    来自 OPENDEV 论文 Lesson 4："LLM 几乎总产出差一点点的内容"。
    从精确匹配开始，逐级放宽条件，最大化编辑成功率。
    """
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    new_text = str(args["new_text"])
    text = path.read_text(encoding="utf-8")

    try:
        result, stage = _fuzzy_replace(text, old_text, new_text)
    except ValueError:
        raise  # 所有阶段都失败，让上层报错

    path.write_text(result, encoding="utf-8")
    stage_names = {1: "exact", 2: "trimmed", 3: "lines", 4: "no_blanks", 5: "anchors", 6: "fuzzy"}
    return f"patched {path.relative_to(context.root)} (stage {stage}: {stage_names.get(stage, 'unknown')})"


def _fuzzy_replace(text, old_text, new_text):
    """6 阶段渐进式匹配，返回 (replaced_text, stage_number)。"""

    lines = text.splitlines(keepends=True)

    # ── Stage 1：精确匹配 ──
    count = text.count(old_text)
    if count == 1:
        return text.replace(old_text, new_text, 1), 1
    if count > 1:
        raise ValueError(f"old_text not unique: found {count} exact matches. "
                         f"Provide more surrounding context to make it unique.")

    # ── Stage 2：去首尾空白 ──
    old_trimmed = old_text.strip()
    if old_trimmed and old_trimmed != old_text:
        count = text.count(old_trimmed)
        if count == 1:
            return text.replace(old_trimmed, new_text.strip(), 1), 2

    # ── Stage 3：逐行匹配（忽略每行首尾空白） ──
    old_lines = [l.strip() for l in old_text.splitlines() if l.strip()]
    if old_lines:
        result = _match_by_lines(lines, old_lines, new_text)
        if result is not None:
            return result, 3

    # ── Stage 4：忽略空白行 ──
    old_nonblank = [l for l in old_text.splitlines(keepends=True) if l.strip()]
    if old_nonblank:
        result = _match_ignore_blanks(text, old_nonblank, new_text)
        if result is not None:
            return result, 4

    # ── Stage 5：上下文锚定（首行 + 末行） ──
    old_line_list = old_text.splitlines()
    if len(old_line_list) >= 2:
        result = _match_by_anchors(lines, old_line_list, new_text)
        if result is not None:
            return result, 5

    # ── Stage 6：首行锚定 + 编辑距离 ──
    if old_line_list:
        result = _match_by_first_line(lines, old_line_list, new_text)
        if result is not None:
            return result, 6

    raise ValueError(
        "Could not match old_text after 6 stages. "
        "Please re-read the file with read_file and try again. "
        "Common causes: the file was modified since you last read it, "
        "or the old_text you provided differs significantly from the actual content."
    )


def _match_by_lines(text_lines, old_stripped_lines, new_text):
    """Stage 3：逐行 strip() 后匹配。"""
    n = len(old_stripped_lines)
    for i in range(len(text_lines) - n + 1):
        window = [l.strip() for l in text_lines[i:i + n]]
        if window == old_stripped_lines:
            before = "".join(text_lines[:i])
            after = "".join(text_lines[i + n:])
            return before + new_text + after
    return None


def _match_ignore_blanks(text, old_nonblank_lines, new_text):
    """Stage 4：忽略空白行后匹配。"""
    old_joined = "".join(old_nonblank_lines)
    all_lines = text.splitlines(keepends=True)
    nonblank = [l for l in all_lines if l.strip()]
    n = len(old_nonblank_lines)
    for i in range(len(nonblank) - n + 1):
        window = "".join(nonblank[i:i + n])
        if window == old_joined:
            # 找到匹配后，在原 text 中定位并替换
            # 用字符级别的索引
            char_idx = 0
            nb_idx = 0
            start_idx = None
            for j, line in enumerate(all_lines):
                if line.strip():
                    if nb_idx == i:
                        start_idx = char_idx
                    nb_idx += 1
                if start_idx is None:
                    char_idx += len(line)
            if start_idx is not None:
                # 从 start_idx 开始，跳过所有非空行找到 n 个非空行
                end_idx = start_idx
                found = 0
                for line in all_lines:
                    if found < n and start_idx <= end_idx:
                        pass
                # 简化：直接在全文级别做替换
                nb_text = "".join(nonblank)
                nb_before = "".join(nonblank[:i])
                nb_after = "".join(nonblank[i + n:])
                # 用 nonblank 定位无法精确还原到原文，回退到近似
                replaced = nb_text.replace("".join(old_nonblank_lines), new_text, 1)
                if replaced != nb_text:
                    # 重建原文：保留空白行，替换非空行区域
                    result_lines = []
                    nb_pos = 0
                    for line in all_lines:
                        if line.strip():
                            if nb_pos == i:
                                result_lines.append(new_text)
                                nb_pos = i + n
                            else:
                                result_lines.append(line)
                                nb_pos += 1
                        else:
                            result_lines.append(line)
                    # 如果 new_text 已经包含了替换行，跳过旧行
                    if nb_pos == i + n:
                        return "".join(result_lines)
                    else:
                        # 回退到更简单的方式
                        pass
    return None


def _match_by_anchors(lines, old_lines, new_text):
    """Stage 5：以首行和末行为锚点，在文件中定位并替换。"""
    first = old_lines[0].strip()
    last = old_lines[-1].strip()
    first_idx = last_idx = None

    for i, line in enumerate(lines):
        if first_idx is None and line.strip() == first:
            first_idx = i
        if line.strip() == last and i >= (first_idx or 0):
            last_idx = i

    if first_idx is not None and last_idx is not None and first_idx <= last_idx:
        before = "".join(lines[:first_idx])
        after = "".join(lines[last_idx + 1:])
        return before + new_text + after
    return None


def _match_by_first_line(lines, old_lines, new_text):
    """Stage 6：仅以首行为锚点，用编辑距离找最佳匹配块。"""
    import difflib
    first = old_lines[0].strip()
    n = len(old_lines)

    best_ratio = 0.6  # 最低相似度阈值
    best_idx = None

    for i, line in enumerate(lines):
        if line.strip() == first or difflib.SequenceMatcher(None, line.strip(), first).ratio() >= best_ratio:
            # 候选锚点，检查后续行的整体相似度
            if i + n <= len(lines):
                window = "".join(l.strip() for l in lines[i:i + n])
                old_joined = "".join(l.strip() for l in old_lines)
                ratio = difflib.SequenceMatcher(None, window, old_joined).ratio()
                if ratio >= best_ratio:
                    best_ratio = ratio
                    best_idx = i

    if best_idx is not None:
        before = "".join(lines[:best_idx])
        after = "".join(lines[best_idx + n:])
        return before + new_text + after
    return None


def tool_file_info(context, args):
    """获取文件元信息：大小、行数、修改时间。"""
    path = context.path(args["path"])
    if not path.exists():
        raise ValueError("path does not exist")
    stat = path.stat()
    line_count = 0
    if path.is_file():
        try:
            line_count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        except Exception:
            line_count = -1
    kind = "file" if path.is_file() else "directory"
    rel = path.relative_to(context.root)
    from datetime import datetime, timezone
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return f"{kind}: {rel}\nsize: {stat.st_size} bytes\nlines: {line_count}\nmodified: {mtime}"


def tool_glob(context, args):
    """按 glob 模式匹配文件路径。"""
    pattern = str(args.get("pattern", "**/*")).strip()
    root = context.path(".")
    import glob as globmod
    matches = []
    for path_str in globmod.glob(pattern, root_dir=root, recursive=True):
        p = root / path_str
        if not p.is_file():
            continue
        if any(part in IGNORED_PATH_NAMES for part in p.relative_to(context.root).parts):
            continue
        matches.append(p.relative_to(context.root).as_posix())
    matches.sort()
    if not matches:
        return "(no matches)"
    result = "\n".join(matches[:200])
    if len(matches) > 200:
        result += f"\n...({len(matches) - 200} more)"
    return result


def tool_grep_count(context, args):
    """统计每个文件的匹配数，不返回匹配内容。"""
    pattern = str(args.get("pattern", "")).strip()
    path = context.path(args.get("path", "."))
    if shutil.which("rg"):
        result = subprocess.run(
            ["rg", "-c", "--no-heading", "--smart-case", pattern, str(path)],
            cwd=context.root, capture_output=True, text=True,
        )
        output = result.stdout.strip()
        if not output:
            return "(no matches)"
        lines = output.splitlines()
        total = 0
        file_count = 0
        for line in lines:
            if ":" in line:
                file_count += 1
                try:
                    total += int(line.rsplit(":", 1)[-1])
                except ValueError:
                    total += 1
        summary = f"{total} matches in {file_count} files"
        if len(lines) <= 20:
            return summary + "\n" + output
        return summary + "\n" + "\n".join(lines[:20]) + f"\n...({len(lines) - 20} more files)"
    # fallback
    total = 0
    files = 0
    search_path = path if path.is_file() else path
    targets = [search_path] if search_path.is_file() else [
        item for item in search_path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(context.root).parts)
    ]
    for file_path in targets:
        try:
            count = file_path.read_text(encoding="utf-8", errors="replace").lower().count(pattern.lower())
            if count > 0:
                files += 1
                total += count
        except Exception:
            continue
    return f"{total} matches in {files} files"


def tool_git_diff(context, args):
    """返回工作区 unified diff。"""
    staged = bool(args.get("staged", False))
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")
    result = subprocess.run(
        cmd, cwd=context.root, capture_output=True, text=True, timeout=15,
    )
    output = result.stdout.strip()
    if not output:
        return "(no changes)"
    return output


def tool_git_log(context, args):
    """返回最近 N 条 git 提交记录。"""
    n = int(args.get("n", 5))
    path = context.path(args.get("path", "."))
    rel = path.relative_to(context.root).as_posix()
    target = "." if rel == "." else rel
    result = subprocess.run(
        ["git", "log", f"-{n}", "--oneline", "--", target],
        cwd=context.root, capture_output=True, text=True, timeout=15,
    )
    output = result.stdout.strip()
    if not output:
        return "(no commits)"
    return output


def tool_delegate(context, args):
    if context.depth >= context.max_depth:
        raise ValueError("delegate depth exceeded")
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")
    return context.spawn_delegate(args)


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
    "file_info": tool_file_info,
    "glob": tool_glob,
    "grep_count": tool_grep_count,
    "git_diff": tool_git_diff,
    "git_log": tool_git_log,
}
