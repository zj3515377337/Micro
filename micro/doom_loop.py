"""死循环检测 —— 在 agent 陷入无效循环时提前介入。

为什么存在：
agent 在长会话中会出现三种典型的坏循环：
1. 简单重复：连续两次调用完全相同的工具（当前 runtime 已能检测）
2. 周期循环：A→B→A→B 这种多拍循环，相邻两次不同但模式重复
3. 探索循环：反复读同一个文件却不做任何修改

OPENDEV 论文发现用 MD5 指纹追踪最近 20 个工具调用、
相同指纹 ≥3 次即注入警告，能减少约 40% 的无效工具消耗。

检测层级（从快到慢、从窄到宽）：
- Layer 1（简单重复）：最近 2 次完全相同 —— 即时拦截
- Layer 2（指纹循环）：最近 20 次中 MD5 相同 ≥3 次 —— 拦截
- Layer 3（重复读取）：同一文件被 read_file ≥4 次 —— 拦截
"""

import hashlib
import json

# 指纹追踪窗口：看最近多少次工具调用
FINGERPRINT_WINDOW = 20
# 相同指纹出现多少次视为死循环
FINGERPRINT_THRESHOLD = 3
# 同一文件被读多少次视为重复读取
READ_REPEAT_THRESHOLD = 4


def tool_fingerprint(name, args):
    """生成工具调用的稳定 MD5 指纹。

    只取 name + args（不取 result），因为这里是在执行*之前*做检测。
    args 按 key 排序后序列化，保证相同语义的调用产生相同指纹。
    """
    key = json.dumps([name, dict(sorted(args.items()))], sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def check_doom_loop(agent, name, args):
    """检查是否存在死循环模式。

    在 ToolExecutor.execute() 中、工具实际执行之前调用。
    如果检测到循环，返回 blocked=True 和一段给模型看的错误消息，
    由 ToolExecutor 将其作为工具执行结果返回（模型会在下一轮看到它）。

    参数：
    - agent: Micro 实例，用于访问 session history
    - name: 即将执行的工具名
    - args: 即将执行的工具参数

    返回值：
    - (blocked: bool, message: str)
      blocked=True 表示应拒绝本次调用，message 是给模型的反馈
    """
    history = agent.session.get("history", [])
    tool_events = [item for item in history if item.get("role") == "tool"]
    recent_window = tool_events[-FINGERPRINT_WINDOW:]

    # ── Layer 1：重复读取同一文件 ──
    # 同一个 path 被 read_file ≥4 次（含本次）→ 拦截。
    # 在 Layer 2（简单重复）之前检查，因为 read 的重复读取消息
    # 比"identical tool call"消息更有指导性。
    if name == "read_file":
        path = str(args.get("path", "")).strip()
        if path:
            read_count = sum(
                1 for item in recent_window
                if item["name"] == "read_file"
                and str(item.get("args", {}).get("path", "")).strip() == path
            )
            if read_count + 1 > READ_REPEAT_THRESHOLD:
                return True, (
                    f"error: you have already read '{path}' {read_count} times. "
                    f"Stop exploring — decide what change to make based on "
                    f"what you've already read, or state clearly what additional "
                    f"information you need and why re-reading would help."
                )

    # ── Layer 2：简单重复 ──
    # 连续两次完全相同的调用 → 立即拦截。
    # 这是最窄也最快的检测，不需要指纹计算。
    if len(tool_events) >= 2:
        recent_two = tool_events[-2:]
        if all(item["name"] == name and item["args"] == args for item in recent_two):
            return True, (
                f"error: repeated identical tool call for {name}; "
                f"choose a different tool or return a final answer"
            )

    # ── Layer 3：MD5 指纹循环 ──
    # 在最近 20 次调用中，相同指纹出现 ≥3 次 → 拦截。
    # 这能捕获简单重复检测不到的 A→B→A→B 周期循环。
    fp = tool_fingerprint(name, args)
    fp_count = sum(
        1 for item in recent_window
        if tool_fingerprint(item["name"], item.get("args", {})) == fp
    )

    # 当前调用算 1 次，和历史上的一起比较阈值
    if fp_count + 1 >= FINGERPRINT_THRESHOLD:
        total = fp_count + 1
        return True, (
            f"error: the same {name} call would be made {total} times "
            f"in the last {FINGERPRINT_WINDOW} tool calls. "
            f"You appear to be stuck in a loop. "
            f"Choose a different approach, or return a final answer explaining "
            f"what you've tried and what you're blocked on."
        )

    return False, ""


def detect_patterns(agent):
    """诊断性接口：返回当前会话中的循环模式摘要。

    不做拦截，只返回统计信息，供 trace/report 使用。
    返回值示例：
    {
        "fingerprint_counts": {"abc123": 3, "def456": 1},
        "read_repeats": {"src/main.py": 5},
        "total_tool_calls": 12,
        "window_size": 20,
    }
    """
    history = agent.session.get("history", [])
    tool_events = [item for item in history if item.get("role") == "tool"]
    recent = tool_events[-FINGERPRINT_WINDOW:]

    fp_counts = {}
    read_counts = {}
    for item in recent:
        fp = tool_fingerprint(item["name"], item.get("args", {}))
        fp_counts[fp] = fp_counts.get(fp, 0) + 1

        if item["name"] == "read_file":
            path = str(item.get("args", {}).get("path", "")).strip()
            if path:
                read_counts[path] = read_counts.get(path, 0) + 1

    return {
        "fingerprint_counts": fp_counts,
        "read_repeats": {path: count for path, count in read_counts.items() if count >= READ_REPEAT_THRESHOLD},
        "total_tool_calls": len(tool_events),
        "window_size": FINGERPRINT_WINDOW,
    }
