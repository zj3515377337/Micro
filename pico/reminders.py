"""事件驱动的 System Reminders —— 在模型注意力衰减时注入定向提醒。

OPENDEV 论文的核心发现：对话超过 15 个工具调用后，模型对初始 system prompt
的注意力明显衰减。解决方案是在**决策点之前**注入简短、聚焦的 `role: user` 提醒。

关键设计：
- 以 `role: user` 注入（非 system），因 user 消息注意力权重更高
- 每类提醒有触发上限，避免变成噪声
- 8 个检测器分三类：pre_model（模型调用前）、post_model（模型输出后）、post_tool（工具执行后）

使用方式：
    from .reminders import ReminderManager
    mgr = ReminderManager(agent)
    reminders = mgr.check("pre_model", tool_steps=3, prompt_metadata={...})
    for text in reminders:
        agent.record({"role": "user", "content": text, "created_at": now()})
"""

# ── 提醒模板 ──────────────────────────────────────────────────────────
# 模板刻意简短（≤3 句），以 `role: user` 的方式注入，像用户直接对模型说话。
# 设计原则来自 OPENDEV §3.2.1：短小、定向、可执行、不做长篇布道。

_REMINDER_TEMPLATES = {
    "exploration_loop": (
        "You have read '{path}' {count} times already. "
        "Stop exploring — decide what change to make based on what you've read, "
        "or state clearly what additional information you need."
    ),
    "too_many_reads": (
        "You have spent {read_ratio_pct}% of {total_turns} turns reading files "
        "without making changes. Consider proposing a concrete action or "
        "returning a final answer with your analysis."
    ),
    "context_pressure": (
        "Context is at {usage_pct}% of capacity. "
        "Prioritize essential actions and avoid unnecessary reads. "
        "If you have enough information to act, do so now."
    ),
    "error_recovery_abandon": (
        "The last {error_count} tool calls returned errors. "
        "Don't give up — check the error messages carefully, "
        "correct the parameters, and try again. "
        "If the approach is fundamentally wrong, switch to a different strategy."
    ),
    "early_claim": (
        "You are returning a final answer after only {tool_steps} tool step(s). "
        "Before concluding, verify: have you inspected all relevant files? "
        "Have you made the necessary changes? If not, continue working."
    ),
    "empty_final": (
        "Your final answer is empty or too short. "
        "Provide a substantive response describing what you did, "
        "what you found, and what the current state is."
    ),
    "premature_stop": (
        "You just executed a tool — review its output carefully before "
        "concluding. The tool result may contain important information "
        "that should inform your next action or final answer."
    ),
    "shell_loop": (
        "The last {cmd_count} tool calls are all shell commands ({sample_cmd}). "
        "You appear to be debugging via trial-and-error without reading source code. "
        "Stop running commands — read the relevant files to understand the root cause, "
        "or return a final answer explaining what you found and what's blocking you."
    ),
    "shell_timeout": (
        "The last {timeout_count} shell commands timed out. "
        "Try a more targeted command with smaller scope, "
        "or increase the timeout value if the operation genuinely needs more time."
    ),
}

# ── 默认触发上限 ──────────────────────────────────────────────────────
# 每类提醒的每会话最大触发次数。达到上限后该检测器静默。
# 来自 OPENDEV 实验：每轮都触发的提醒会变成噪声被模型学会忽略。

DEFAULT_MAX_FIRES = {
    "exploration_loop": 2,
    "too_many_reads": 1,
    "context_pressure": 999,  # 每轮都允许（但实际每轮最多触发 1 次，由 phase 控制）
    "error_recovery_abandon": 3,
    "shell_loop": 2,
    "early_claim": 2,
    "empty_final": 2,
    "premature_stop": 2,
    "shell_timeout": 2,
}

# 上下文压力阈值（prompt 使用比例）
CONTEXT_PRESSURE_THRESHOLD = 0.85

# "过多读取"阈值
TOO_MANY_READS_RATIO = 0.55
TOO_MANY_READS_MIN_TURNS = 4

# "过早声称完成"阈值：tool_steps 为 0 且 max_steps >= 2 才触发。
# 短任务（max_steps=1）允许直接回答无需提醒。
EARLY_CLAIM_MIN_STEPS = 0

# early_claim 仅在用户消息包含"行动关键词"时触发（避免对纯问答触发）
_EARLY_CLAIM_ACTION_KEYWORDS = (
    "fix", "change", "create", "write", "modify", "add", "remove",
    "update", "implement", "refactor", "build", "patch", "edit",
    "修改", "创建", "添加", "修复", "重构", "实现", "更改",
)

# "空 final" 最小字符数：低于此值认为模型没有认真回答
EMPTY_FINAL_MIN_CHARS = 5

# "连续错误"计数阈值
CONSECUTIVE_ERROR_THRESHOLD = 2


def _count_recent_reads(agent):
    """统计最近一次 read_file 被重复读取的情况。"""
    history = agent.session.get("history", [])
    tool_events = [item for item in history if item.get("role") == "tool"]
    # 找最近一次 read_file 被读了几次
    recent_reads = [
        item for item in tool_events
        if item["name"] == "read_file"
    ]
    if not recent_reads:
        return None, 0

    last_path = str(recent_reads[-1].get("args", {}).get("path", "")).strip()
    if not last_path:
        return None, 0

    count = sum(
        1 for item in recent_reads
        if str(item.get("args", {}).get("path", "")).strip() == last_path
    )
    return last_path, count


def _count_recent_errors(agent):
    """统计最近连续的工具错误次数。

    包括两种：
    1. 工具本身报错（以 error: 开头）
    2. shell 命令返回非零退出码
    """
    history = agent.session.get("history", [])
    tool_events = [item for item in history if item.get("role") == "tool"]
    count = 0
    for item in reversed(tool_events):
        content = str(item.get("content", ""))
        is_error = content.startswith("error:")
        is_shell_fail = (
            item.get("name") == "run_shell"
            and "exit_code:" in content
            and "exit_code: 0" not in content
        )
        if is_error or is_shell_fail:
            count += 1
        else:
            break
    return count


def _count_shell_timeouts(agent):
    """统计最近连续的 shell 超时次数。"""
    history = agent.session.get("history", [])
    tool_events = [item for item in history if item.get("role") == "tool"]
    count = 0
    for item in reversed(tool_events):
        if item["name"] == "run_shell":
            content = str(item.get("content", ""))
            if "timed out" in content.lower() or "timeout" in content.lower():
                count += 1
            else:
                break
        else:
            break
    return count


def _count_shell_loop(agent):
    """检测连续跑 shell 命令的死循环。

    返回 (连续 shell 命令数, 示例命令)。
    只在"全是 shell，没有 read_file"时才触发——说明模型在盲目试错。
    """
    history = agent.session.get("history", [])
    tool_events = [item for item in history if item.get("role") == "tool"]
    count = 0
    sample = ""
    for item in reversed(tool_events):
        if item.get("name") == "run_shell":
            count += 1
            if not sample:
                cmd = str(item.get("args", {}).get("command", ""))[:60]
                sample = cmd
        elif item.get("name") in ("read_file", "list_files", "search"):
            break
        else:
            break
    return count, sample


def _read_ratio(agent):
    """计算最近工具调用中 read_file 的占比。"""
    history = agent.session.get("history", [])
    tool_events = [item for item in history if item.get("role") == "tool"]
    if not tool_events:
        return 0.0, 0
    reads = sum(1 for item in tool_events if item["name"] == "read_file")
    return reads / len(tool_events), len(tool_events)


class ReminderManager:
    """管理 8 个事件检测器，按 phase 分发检查并返回提醒文本列表。"""

    def __init__(self, agent):
        self.agent = agent
        self._counters = {}  # detector_name -> int (当前会话已触发次数)

    def _first_user_message(self):
        """获取会话的第一条用户消息（原始请求）。"""
        history = self.agent.session.get("history", [])
        for item in history:
            if item.get("role") == "user":
                return str(item.get("content", ""))
        return ""

    def _can_fire(self, name):
        """检查是否还能触发（未达到上限）。"""
        max_fires = DEFAULT_MAX_FIRES.get(name, 1)
        return self._counters.get(name, 0) < max_fires

    def _record_fire(self, name):
        self._counters[name] = self._counters.get(name, 0) + 1

    def check(self, phase, **ctx):
        """运行指定 phase 的所有检测器，返回提醒文本列表。

        参数：
        - phase: "pre_model" | "post_model" | "post_tool"
        - **ctx: 各检测器需要的上下文（tool_steps, attempts, prompt_metadata 等）

        返回：list[str]，可能为空。
        """
        if phase == "pre_model":
            return self._check_pre_model(**ctx)
        elif phase == "post_model":
            return self._check_post_model(**ctx)
        elif phase == "post_tool":
            return self._check_post_tool(**ctx)
        return []

    # ── pre_model 检测器 ───────────────────────────────────────────

    def _check_pre_model(self, tool_steps=0, attempts=0, prompt_metadata=None, **__):
        reminders = []

        # 1. exploration_loop
        path, count = _count_recent_reads(self.agent)
        if path and count >= 3 and self._can_fire("exploration_loop"):
            reminders.append(
                _REMINDER_TEMPLATES["exploration_loop"].format(path=path, count=count)
            )
            self._record_fire("exploration_loop")

        # 2. too_many_reads
        ratio, total = _read_ratio(self.agent)
        if ratio > TOO_MANY_READS_RATIO and total >= TOO_MANY_READS_MIN_TURNS and self._can_fire("too_many_reads"):
            reminders.append(
                _REMINDER_TEMPLATES["too_many_reads"].format(
                    read_ratio_pct=int(ratio * 100), total_turns=total
                )
            )
            self._record_fire("too_many_reads")

        # 3. context_pressure
        if prompt_metadata and self._can_fire("context_pressure"):
            prompt_chars = int(prompt_metadata.get("prompt_chars", 0))
            budget_chars = int(prompt_metadata.get("prompt_budget_chars", 12000))
            if budget_chars > 0 and prompt_chars / budget_chars >= CONTEXT_PRESSURE_THRESHOLD:
                reminders.append(
                    _REMINDER_TEMPLATES["context_pressure"].format(
                        usage_pct=int(prompt_chars / budget_chars * 100)
                    )
                )
                self._record_fire("context_pressure")

        # 4. error_recovery_abandon
        error_count = _count_recent_errors(self.agent)
        if error_count >= CONSECUTIVE_ERROR_THRESHOLD and self._can_fire("error_recovery_abandon"):
            reminders.append(
                _REMINDER_TEMPLATES["error_recovery_abandon"].format(error_count=error_count)
            )
            self._record_fire("error_recovery_abandon")

        # 5. shell_loop：连续跑 shell 命令排查环境问题（trial-and-error 死循环）
        cmd_count, sample_cmd = _count_shell_loop(self.agent)
        if cmd_count >= 4 and self._can_fire("shell_loop"):
            reminders.append(
                _REMINDER_TEMPLATES["shell_loop"].format(cmd_count=cmd_count, sample_cmd=sample_cmd)
            )
            self._record_fire("shell_loop")

        return reminders

    # ── post_model 检测器 ──────────────────────────────────────────

    def _check_post_model(self, tool_steps=0, kind="", payload=None, **__):
        reminders = []

        if kind != "final":
            return reminders

        final_text = str(payload or "").strip()

        # 5. early_claim：仅当用户请求包含行动关键词、且模型没用工具时触发。
        if (tool_steps <= EARLY_CLAIM_MIN_STEPS
                and self.agent.max_steps >= 2
                and self._can_fire("early_claim")):
            user_msg = self._first_user_message().lower()
            if any(kw in user_msg for kw in _EARLY_CLAIM_ACTION_KEYWORDS):
                reminders.append(
                    _REMINDER_TEMPLATES["early_claim"].format(tool_steps=tool_steps)
                )
                self._record_fire("early_claim")

        # 6. empty_final
        if len(final_text) < EMPTY_FINAL_MIN_CHARS and self._can_fire("empty_final"):
            reminders.append(_REMINDER_TEMPLATES["empty_final"])
            self._record_fire("empty_final")

        # 7. premature_stop：仅当上一个工具执行失败/部分成功时才触发。
        # 正常的 tool→final 流程不应该被拦截（模型已经审阅了工具输出）。
        history = self.agent.session.get("history", [])
        last_is_tool = history and history[-1].get("role") == "tool"
        if last_is_tool and self._can_fire("premature_stop"):
            last_content = str(history[-1].get("content", "")).lower()
            has_error = "error:" in last_content or "failed" in last_content or "traceback" in last_content
            if has_error:
                reminders.append(_REMINDER_TEMPLATES["premature_stop"])
                self._record_fire("premature_stop")

        return reminders

    # ── post_tool 检测器 ───────────────────────────────────────────

    def _check_post_tool(self, name="", metadata=None, **__):
        reminders = []

        if name != "run_shell":
            return reminders

        # 8. shell_timeout
        timeout_count = _count_shell_timeouts(self.agent)
        if timeout_count >= CONSECUTIVE_ERROR_THRESHOLD and self._can_fire("shell_timeout"):
            reminders.append(
                _REMINDER_TEMPLATES["shell_timeout"].format(timeout_count=timeout_count)
            )
            self._record_fire("shell_timeout")

        return reminders
