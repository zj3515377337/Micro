"""Plan Mode —— 先规划、审查、再执行的交互式工作流。

OPENDEV §3.4 的双智能体架构：
- Planner agent：只读工具（read_file / list_files / search），探索代码库
- 输出 7 节结构化计划
- 用户审查后注入 prompt，Normal Mode agent 按计划执行

两种进入方式：
1. 显式：用户输入 /plan "task"
2. 自动：thinking 模型判断复杂任务需要先规划
"""

from pathlib import Path

from .workspace import now

PLANNER_SYSTEM_PROMPT = (
    "You are a strategic planner analyzing a codebase to create a detailed plan. "
    "You have READ-ONLY access — you can read files, search code, and list directories. "
    "Your goal is to UNDERSTAND the relevant code and produce a clear, actionable plan.\n\n"
    "After exploring, output your plan in this EXACT format:\n\n"
    "## Goal\n"
    "[One sentence describing what will be accomplished]\n\n"
    "## Context\n"
    "[Key findings from your exploration — relevant files, patterns, constraints]\n\n"
    "## Files to modify\n"
    "- path/to/file1.tsx — [what needs to change]\n"
    "- path/to/file2.ts  — [what needs to change]\n\n"
    "## New files (if any)\n"
    "- path/to/new/file  — [what this file does]\n"
    "(or 'None')\n\n"
    "## Steps\n"
    "1. [First action — be specific: which file, what to do]\n"
    "2. [Second action]\n"
    "...\n\n"
    "## Verification\n"
    "- [How to verify the change works]\n"
    "- [Test command or manual check]\n\n"
    "## Risks\n"
    "- [Potential issue or pitfall]\n"
    "(or 'None identified')\n\n"
    "End your response with exactly the plan above. "
    "Do NOT include any conversational text before or after the plan structure."
)


def generate_plan(agent, user_request):
    """用只读 Planner agent 探索代码库，返回结构化计划（markdown 字符串）。

    参数：
    - agent: 父 Micro 实例（提供 model_client、workspace、session_store）
    - user_request: 用户的任务描述

    返回：plan_text (str)
    """
    # 构造只读 Planner agent
    # planner_model_client 如果配置了，同时用于 action 和 thinking
    dedicated = getattr(agent, "planner_model_client", None)
    action_model = dedicated or agent.model_client
    thinking_model = dedicated or getattr(agent, "thinking_model_client", None)
    planner = agent.__class__(
        model_client=action_model,
        thinking_model_client=thinking_model,
        workspace=agent.workspace,
        session_store=agent.session_store,
        run_store=agent.run_store,
        approval_policy="auto",    # 只读，无需审批
        max_steps=5,
        max_new_tokens=agent.max_new_tokens,
        depth=agent.depth,
        max_depth=agent.max_depth,
        read_only=False,           # 用 allowed_tools 控制权限，而非 read_only
        allowed_tools=["read_file", "list_files", "search", "delegate"],
        secret_env_names=agent.secret_env_names,
        shell_env_allowlist=agent.shell_env_allowlist,
    )

    # 让 Planner 探索并生成计划
    planner_prompt = (
        f"Explore the codebase to understand how to accomplish this task:\n\n"
        f"{user_request}\n\n"
        f"Read the relevant files, search for patterns, understand the current "
        f"architecture. Then output a structured plan using the format specified "
        f"in your instructions."
    )
    result = planner.ask(planner_prompt)

    # 提取计划（去掉模型可能附加的对话文本）
    plan_text = _extract_plan_from_output(str(result))
    if not plan_text:
        plan_text = str(result)  # 提取失败则用原始输出

    return plan_text


def _extract_plan_from_output(raw_output):
    """从模型输出中提取计划部分（以 ## Goal 开头到末尾）。"""
    text = str(raw_output).strip()
    goal_idx = text.find("## Goal")
    if goal_idx == -1:
        return ""
    return text[goal_idx:].strip()


def save_plan(workspace_root, plan_text, task_summary=""):
    """保存计划到 .pico/plans/ 目录，返回文件路径。"""
    root = Path(workspace_root)
    plans_dir = root / ".pico" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now()[:19].replace(":", "").replace("T", "-")
    slug = _slugify(str(task_summary)[:40]) or "plan"
    filename = f"plan_{timestamp}_{slug}.md"
    plan_path = plans_dir / filename
    plan_path.write_text(plan_text, encoding="utf-8")
    return plan_path


def _slugify(text):
    """将文本转为文件名友好的 slug。"""
    import re
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text[:50]


def latest_plan_text(workspace_root):
    """读取最新的计划文件内容（按文件名排序）。"""
    root = Path(workspace_root)
    plans_dir = root / ".pico" / "plans"
    if not plans_dir.exists():
        return ""
    plans = sorted(plans_dir.glob("plan_*.md"), reverse=True)
    if not plans:
        return ""
    return plans[0].read_text(encoding="utf-8").strip()


def clear_plans(workspace_root):
    """清除所有计划文件。"""
    root = Path(workspace_root)
    plans_dir = root / ".pico" / "plans"
    if plans_dir.exists():
        for f in plans_dir.glob("plan_*.md"):
            f.unlink()
