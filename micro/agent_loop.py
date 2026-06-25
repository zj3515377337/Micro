"""Agent control loop extracted from the runtime facade."""

import time

from .checkpoint import CHECKPOINT_NONE_STATUS, CHECKPOINT_PARTIAL_STALE_STATUS, CHECKPOINT_WORKSPACE_MISMATCH_STATUS
from .task_state import TaskState
from .workspace import clip, now


def _build_thinking_prompt(full_prompt, user_message):
    """为 thinking 模型构建纯推理 prompt（不含工具 schema）。

    thinking 模型只需要理解当前状态并给出分析，不需要知道有哪些工具。
    输出是自由文本，不做结构化解析。
    """
    user_request = str(user_message).strip()
    return (
        f"You are an expert software engineer analyzing a coding task. "
        f"Think through the situation carefully — what is the user asking for, "
        f"what is the current state of the repository, and what approach should be taken?\n\n"
        f"Context (workspace, memory, conversation history):\n{full_prompt}\n\n"
        f"Analyze the situation above and outline a clear plan of action for: {user_request}\n\n"
        f"Your analysis (be specific about which files to examine, what to look for, "
        f"and the sequence of steps):"
    )


def _build_critique_prompt(thinking_output, user_message, history):
    """为 critique 阶段构建审查 prompt。

    审查 thinking 输出是否合理：计划有遗漏吗？上一步工具结果被正确解读了吗？
    输出简短的结构化反馈。
    """
    user_request = str(user_message).strip()
    # 提取最近 3 个工具结果作为上下文
    tool_results = []
    for item in reversed(history):
        if item.get("role") == "tool":
            content = str(item.get("content", ""))[:300]
            tool_results.append(f"[{item.get('name', 'tool')}] {content}")
            if len(tool_results) >= 3:
                break
    recent_tools = "\n".join(reversed(tool_results)) if tool_results else "(none)"

    return (
        f"You are reviewing another agent's analysis for a coding task. "
        f"Be critical — identify gaps, risks, and misinterpretations.\n\n"
        f"Original request: {user_request}\n\n"
        f"Recent tool results:\n{recent_tools}\n\n"
        f"The agent's analysis:\n{thinking_output}\n\n"
        f"Evaluate the analysis:\n"
        f"- Does the plan address the user's request?\n"
        f"- Are there gaps or missing steps?\n"
        f"- Were the recent tool results correctly interpreted?\n"
        f"- Is there a simpler or more reliable approach?\n\n"
        f"Provide a SHORT critique (2-4 sentences). If the plan looks good, say so. "
        f"If there are issues, point them out specifically."
    )


class AgentLoop:
    def __init__(self, agent):
        self.agent = agent

    def run(self, user_message):
        agent = self.agent
        run_started_at = time.monotonic()
        agent.memory.set_task_summary(user_message)
        agent.record({"role": "user", "content": user_message, "created_at": now()})

        task_state = TaskState.create(run_id=agent.new_run_id(), task_id=agent.new_task_id(), user_request=user_message)
        task_state.resume_status = agent.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        agent.current_task_state = task_state
        agent.current_run_dir = agent.run_store.start_run(task_state)
        agent.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )

        tool_steps = 0
        attempts = 0
        max_attempts = max(agent.max_steps * 3, agent.max_steps + 4)

        # 这是 agent 的主循环，可以按“感知 -> 决策 -> 行动 -> 记录”来理解：
        # 1. 感知：重新组 prompt，把当前状态整理给模型看
        # 2. 决策：让模型返回一个工具调用，或一个最终答案
        # 3. 行动：如果是工具调用，就执行工具
        # 4. 记录：把结果写回 history / task_state / trace / memory
        # 然后进入下一轮，直到停机条件满足
        while tool_steps < agent.max_steps and attempts < max_attempts:
            attempts += 1
            task_state.record_attempt()
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            prompt, prompt_metadata = agent._build_prompt_and_metadata(user_message)
            agent.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="freshness_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",
                    },
                )
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                agent.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="workspace_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            if prompt_metadata.get("budget_reductions"):
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="context_reduction")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )

            # ★ 注入点 #1：pre_model 提醒（在 prompt 组好后、模型调用前）
            pre_reminders = agent.check_reminders(
                "pre_model",
                tool_steps=tool_steps,
                attempts=attempts,
                prompt_metadata=prompt_metadata,
            )
            for reminder_text in pre_reminders:
                agent.record({"role": "user", "content": reminder_text, "created_at": now()})
                agent.emit_trace(
                    task_state,
                    "reminder_injected",
                    {
                        "phase": "pre_model",
                        "reminder": reminder_text,
                    },
                )

            # ★ Thinking 阶段：用独立模型做纯推理（无工具压力），输出作为 action 模型的上下文
            thinking_client = getattr(agent, "thinking_model_client", None)
            if thinking_client is not None and agent.max_steps >= 4:
                thinking_started_at = time.monotonic()
                try:
                    thinking_prompt = _build_thinking_prompt(prompt, user_message)
                    thinking_output = thinking_client.complete(thinking_prompt, min(agent.max_new_tokens, 1024))
                    thinking_text = str(thinking_output).strip()
                    if thinking_text:
                        agent.record({"role": "user", "content": f"[Thinking]\n{thinking_text}", "created_at": now()})
                        agent.emit_trace(
                            task_state,
                            "thinking_completed",
                            {
                                "thinking_chars": len(thinking_text),
                                "duration_ms": int((time.monotonic() - thinking_started_at) * 1000),
                            },
                        )

                        # ★ Self-Critique 阶段：用同一模型审视 thinking 输出 + 最近工具结果
                        if agent.max_steps >= 6:
                            try:
                                critique_prompt = _build_critique_prompt(
                                    thinking_text, user_message, agent.session.get("history", [])
                                )
                                critique_output = thinking_client.complete(critique_prompt, min(agent.max_new_tokens, 512))
                                critique_text = str(critique_output).strip()
                                if critique_text:
                                    agent.record({"role": "user", "content": f"[Critique]\n{critique_text}", "created_at": now()})
                                    agent.emit_trace(
                                        task_state,
                                        "critique_completed",
                                        {
                                            "critique_chars": len(critique_text),
                                        },
                                    )
                            except Exception:
                                pass

                except Exception as exc:
                    agent.emit_trace(task_state, "thinking_failed", {"error": str(exc)[:200]})

            agent.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            prompt_cache_key = None
            prompt_cache_retention = None
            if getattr(agent.model_client, "supports_prompt_cache", False):
                # 只有后端明确支持时，才把稳定前缀的 hash 作为 cache key 发出去。
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()
            raw = agent.model_client.complete(
                prompt,
                agent.max_new_tokens,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                # 把后端返回的 usage/cache 统计并回 prompt_metadata，
                # 方便统一写入 report 和 trace。
                prompt_metadata.update(completion_metadata)
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata
            kind, payload = agent.parse(raw)
            agent.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                task_state.record_tool(name)
                tool_started_at = time.monotonic()
                tool_result = agent.execute_tool(name, args)
                result = tool_result.content
                agent.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": name,
                        "args": args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        **dict(tool_result.metadata or {}),
                    },
                )

                # ★ 注入点 #3：post_tool 提醒（工具执行后）
                post_tool_reminders = agent.check_reminders(
                    "post_tool",
                    tool_steps=tool_steps,
                    name=name,
                    metadata=dict(tool_result.metadata or {}),
                )
                for reminder_text in post_tool_reminders:
                    agent.record({"role": "user", "content": reminder_text, "created_at": now()})
                    agent.emit_trace(
                        task_state,
                        "reminder_injected",
                        {
                            "phase": "post_tool",
                            "reminder": reminder_text,
                        },
                    )

                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="tool_executed")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
                continue

            if kind == "retry":
                agent.record({"role": "assistant", "content": payload, "created_at": now()})
                agent.run_store.write_task_state(task_state)
                continue

            # kind == "final"：模型认为可以结束了
            final = (payload or raw).strip()

            # ★ 注入点 #2：post_model 提醒（模型返回 final 后、正式接受前）
            post_reminders = agent.check_reminders(
                "post_model",
                tool_steps=tool_steps,
                kind="final",
                payload=final,
            )
            if post_reminders:
                # 把模型的 final 当作 assistant 消息记录（类似 retry），
                # 注入提醒，然后回到循环开头让模型再试一次。
                agent.record({"role": "assistant", "content": final, "created_at": now()})
                for reminder_text in post_reminders:
                    agent.record({"role": "user", "content": reminder_text, "created_at": now()})
                    agent.emit_trace(
                        task_state,
                        "reminder_injected",
                        {
                            "phase": "post_model",
                            "reminder": reminder_text,
                        },
                    )
                agent.run_store.write_task_state(task_state)
                continue

            agent.record({"role": "assistant", "content": final, "created_at": now()})
            task_state.finish_success(final)
            agent.promote_durable_memory(user_message, final)
            checkpoint = agent.create_checkpoint(task_state, user_message, trigger="run_finished")
            agent.run_store.write_task_state(task_state)
            agent.emit_trace(
                task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trigger": "run_finished",
                },
            )
            agent.emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
            return final

        if attempts >= max_attempts and tool_steps < agent.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
        else:
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        agent.record({"role": "assistant", "content": final, "created_at": now()})
        agent.promote_durable_memory(user_message, final)
        agent.run_store.write_task_state(task_state)
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
        return final
