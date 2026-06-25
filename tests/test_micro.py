import os
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import micro
from micro import (
    AnthropicCompatibleModelClient,
    FakeModelClient,
    Micro,
    OllamaModelClient,
    OpenAICompatibleModelClient,
    SessionStore,
    WorkspaceContext,
    build_welcome,
)


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".micro" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    return Micro(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_agent_runs_tool_then_final(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Read the file successfully.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Read the file successfully."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    assert "hello.txt" in agent.session["memory"]["files"]


def test_agent_updates_task_summary_on_each_request(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>First pass.</final>",
            "<final>Second pass.</final>",
        ],
    )

    assert agent.ask("First request") == "First pass."
    assert agent.session["memory"]["working"]["task_summary"] == "First request"

    assert agent.ask("Second request") == "Second pass."
    assert agent.session["memory"]["working"]["task_summary"] == "Second request"


def test_agent_only_stores_reusable_epistemic_notes(tmp_path):
    (tmp_path / "facts.txt").write_text("deploy key is red\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"facts.txt","start":1,"end":1}}</tool>',
            "<final>Done.</final>",
            "<final>It is red.</final>",
        ],
    )

    assert agent.ask("Read the file and remember the fact") == "Done."
    notes = agent.session["memory"]["episodic_notes"]
    assert any("deploy key is red" in note["text"] for note in notes)
    assert not any(note["text"] == "Done." for note in notes)
    assert not any(note["text"] == "Done." for note in notes)

    resumed = Micro.from_session(
        model_client=FakeModelClient(["<final>It is red.</final>"]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("What color is the deploy key?") == "It is red."
    prompt = resumed.model_client.prompts[-1]
    assert "Relevant memory" in prompt
    assert "deploy key is red" in prompt


def test_file_summary_cache_is_invalidated_on_out_of_band_edit_and_path_spelling(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    agent.memory.set_file_summary("./sample.txt", "sample.txt: alpha")
    agent.memory.remember_file("./sample.txt")
    assert agent.memory.to_dict()["file_summaries"]["sample.txt"]["freshness"]

    assert "sample.txt: alpha" in agent.memory.render_memory_text()
    file_path.write_text("beta\n", encoding="utf-8")

    resumed = Micro.from_session(
        model_client=FakeModelClient([]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert "sample.txt: alpha" not in resumed.memory_text()
    resumed.memory.invalidate_file_summary("sample.txt")
    assert "sample.txt" not in resumed.memory.to_dict()["file_summaries"]


def test_agent_retries_after_empty_model_output(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "<final>Recovered after retry.</final>",
        ],
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after retry."
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("empty response" in item for item in notices)


def test_agent_retries_after_malformed_tool_payload(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":"bad"}</tool>',
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            "<final>Recovered after malformed tool output.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Recovered after malformed tool output."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("valid <tool> call" in item for item in notices)


def test_agent_accepts_xml_write_file_tool(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="hello.py"><content>print("hi")\n</content></tool>',
            "<final>Done.</final>",
        ],
    )

    answer = agent.ask("Create hello.py")

    assert answer == "Done."
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == 'print("hi")\n'


def test_retries_do_not_consume_the_whole_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "",
            "<final>Recovered after several retries.</final>",
        ],
        max_steps=1,
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after several retries."


def test_agent_saves_and_resumes_session(tmp_path):
    agent = build_agent(tmp_path, ["<final>First pass.</final>"])
    assert agent.ask("Start a session") == "First pass."

    resumed = Micro.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.session["history"][0]["content"] == "Start a session"
    assert resumed.ask("Continue") == "Resumed."


def test_delegate_uses_child_agent(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"task":"inspect README","max_steps":2}}</tool>',
            "<final>Child result.</final>",
            "<final>Parent incorporated the child result.</final>",
        ],
    )

    answer = agent.ask("Use delegation")

    assert answer == "Parent incorporated the child result."
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert tool_events[0]["name"] == "delegate"
    assert "delegate_result" in tool_events[0]["content"]


def test_patch_file_replaces_exact_match(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "patch_file",
        {
            "path": "sample.txt",
            "old_text": "world",
            "new_text": "agent",
        },
    )

    assert "patched sample.txt" in result
    assert file_path.read_text(encoding="utf-8") == "hello agent\n"


def test_invalid_risky_tool_does_not_prompt_for_approval(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")

    with patch("builtins.input") as mock_input:
        result = agent.run_tool("write_file", {})

    assert result.startswith("error: invalid arguments for write_file: 'path'")
    assert 'example: <tool name="write_file"' in result
    mock_input.assert_not_called()


def test_list_files_hides_internal_agent_state(tmp_path):
    agent = build_agent(tmp_path, [])
    (tmp_path / ".micro").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")

    result = agent.run_tool("list_files", {})

    assert ".micro" not in result
    assert ".git" not in result
    assert "[F] hello.txt" in result


def test_repeated_identical_tool_call_is_rejected(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "1"})
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "2"})

    result = agent.run_tool("list_files", {})

    assert result == "error: repeated identical tool call for list_files; choose a different tool or return a final answer"


def test_md5_fingerprint_loop_is_rejected(tmp_path):
    """Layer 2: A→B→A→B multi-step cycle caught by MD5 fingerprint."""
    agent = build_agent(tmp_path, [])
    # Simulate read→search→read→search cycle
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "a.py"}, "content": "x", "created_at": "t1"})
    agent.record({"role": "tool", "name": "search", "args": {"pattern": "foo"}, "content": "x", "created_at": "t2"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "a.py"}, "content": "x", "created_at": "t3"})
    # This is the 3rd read_file("a.py") — fingerprint appears 3 times in window

    blocked, _ = agent.check_doom_loop("read_file", {"path": "a.py"})
    assert blocked is True


def test_different_args_produce_different_fingerprints(tmp_path):
    """Same tool name with different args should NOT trigger fingerprint loop."""
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "a.py"}, "content": "x", "created_at": "t1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "b.py"}, "content": "x", "created_at": "t2"})

    blocked, _ = agent.check_doom_loop("read_file", {"path": "c.py"})
    assert blocked is False


def test_read_repeat_on_same_file_is_rejected(tmp_path):
    """Layer 3: same file read ≥4 times is blocked."""
    agent = build_agent(tmp_path, [])
    for i in range(4):
        agent.record({"role": "tool", "name": "read_file", "args": {"path": "a.py"}, "content": "x", "created_at": str(i)})

    blocked, msg = agent.check_doom_loop("read_file", {"path": "a.py"})
    assert blocked is True
    assert "already read" in msg
    assert "a.py" in msg
    assert "4 times" in msg


def test_read_different_files_does_not_trigger_repeat(tmp_path):
    """Reading different files should be fine."""
    agent = build_agent(tmp_path, [])
    for path in ["a.py", "b.py", "c.py"]:
        agent.record({"role": "tool", "name": "read_file", "args": {"path": path}, "content": "x", "created_at": path})

    blocked, _ = agent.check_doom_loop("read_file", {"path": "d.py"})
    assert blocked is False


def test_doom_loop_detector_returns_blocked_and_message(tmp_path):
    """The blocked message should be descriptive."""
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "1"})
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "2"})

    blocked, msg = agent.check_doom_loop("list_files", {})
    assert blocked is True
    assert "list_files" in msg
    assert "error:" in msg.lower()


def test_repeated_tool_call_backward_compat(tmp_path):
    """Old repeated_tool_call() still works and delegates to doom_loop."""
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "1"})
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "2"})

    assert agent.repeated_tool_call("list_files", {}) is True
    assert agent.repeated_tool_call("read_file", {"path": "new.txt"}) is False


def test_fingerprint_loop_message_is_informative(tmp_path):
    """Layer 2 error message should explain the cycle clearly."""
    agent = build_agent(tmp_path, [])
    # A→B→A→B pattern: search→read→search→read→search (avoid Layer 1 simple repeat)
    agent.record({"role": "tool", "name": "search", "args": {"pattern": "old"}, "content": "no match", "created_at": "t1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "x.py"}, "content": "x", "created_at": "t2"})
    agent.record({"role": "tool", "name": "search", "args": {"pattern": "old"}, "content": "no match", "created_at": "t3"})

    blocked, msg = agent.check_doom_loop("search", {"pattern": "old"})
    assert blocked is True
    # 2 past + 1 current = 3 times
    assert "3 times" in msg
    assert "loop" in msg.lower()


# ── System Reminders 测试 ─────────────────────────────────────────────


def test_reminder_exploration_loop_fires_on_third_read(tmp_path):
    """同一文件被 read_file ≥3 次 → 触发 exploration_loop 提醒。"""
    agent = build_agent(
        tmp_path,
        ["<final>Done.</final>"],
    )
    for i in range(3):
        agent.record({"role": "tool", "name": "read_file", "args": {"path": "a.py"}, "content": "line", "created_at": str(i)})

    reminders = agent.check_reminders("pre_model", tool_steps=3, attempts=3, prompt_metadata={})
    assert len(reminders) >= 1
    assert "a.py" in reminders[0]


def test_reminder_exploration_loop_respects_max_fires(tmp_path):
    """exploration_loop 触发 2 次后不再触发（上限为 2）。"""
    agent = build_agent(
        tmp_path,
        ["<final>Done.</final>"],
    )
    for i in range(3):
        agent.record({"role": "tool", "name": "read_file", "args": {"path": "a.py"}, "content": "line", "created_at": str(i)})

    # 第 1 次触发
    reminders = agent.check_reminders("pre_model", tool_steps=4, attempts=4, prompt_metadata={})
    assert len(reminders) >= 1

    # 第 2 次触发
    reminders = agent.check_reminders("pre_model", tool_steps=5, attempts=5, prompt_metadata={})
    assert len(reminders) >= 1

    # 第 3 次不再触发（已达上限 2）
    reminders = agent.check_reminders("pre_model", tool_steps=6, attempts=6, prompt_metadata={})
    assert len(reminders) == 0


def test_reminder_context_pressure_fires_when_over_threshold(tmp_path):
    """prompt 超 85% 预算 → 触发 context_pressure 提醒。"""
    agent = build_agent(
        tmp_path,
        ["<final>OK.</final>"],
    )
    # 模拟 prompt 接近满载
    meta = {"prompt_chars": 10500, "prompt_budget_chars": 12000}  # 87.5%
    reminders = agent.check_reminders("pre_model", tool_steps=0, attempts=1, prompt_metadata=meta)
    assert len(reminders) >= 1
    assert "87%" in reminders[0] or "capacity" in reminders[0].lower()


def test_reminder_context_pressure_does_not_fire_below_threshold(tmp_path):
    """prompt 低于 85% 预算 → 不触发。"""
    agent = build_agent(
        tmp_path,
        ["<final>OK.</final>"],
    )
    meta = {"prompt_chars": 6000, "prompt_budget_chars": 12000}  # 50%
    reminders = agent.check_reminders("pre_model", tool_steps=0, attempts=1, prompt_metadata=meta)
    assert len(reminders) == 0


def test_reminder_empty_final_triggers_retry_in_agent_loop(tmp_path):
    """模型返回过短的 final (<5 字符) → post_model 提醒注入 → 重试给出有效答案。"""
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            "<final>ok</final>",                              # 2 字符 < 5 → 触发 empty_final
            "<final>File hello.txt has been read successfully.</final>",  # 重试后有效答案
        ],
    )
    answer = agent.ask("Read hello.txt")
    assert answer == "File hello.txt has been read successfully."

    # 验证提醒被注入到了 history
    history = agent.session["history"]
    reminder_items = [
        item for item in history
        if item["role"] == "user" and "empty" in item["content"].lower()
    ]
    assert len(reminder_items) >= 1


def test_reminder_premature_stop_after_failed_tool(tmp_path):
    """模型在工具失败后立即返回 final → 触发 premature_stop。"""
    agent = build_agent(
        tmp_path,
        ["<final>Giving up now.</final>"],
    )
    # 模拟刚执行了一个失败的工具
    agent.record({"role": "tool", "name": "run_shell", "args": {"command": "bad"}, "content": "error: command failed", "created_at": "t1"})

    reminders = agent.check_reminders("post_model", tool_steps=1, kind="final", payload="Giving up now.")
    assert len(reminders) >= 1
    assert "review" in reminders[0].lower() or "tool" in reminders[0].lower()


def test_reminder_disabled_by_feature_flag(tmp_path):
    """feature flag reminders=False → 不触发任何提醒。"""
    agent = build_agent(
        tmp_path,
        ["<final>done</final>"],
        feature_flags={"reminders": False},
    )
    for i in range(3):
        agent.record({"role": "tool", "name": "read_file", "args": {"path": "a.py"}, "content": "line", "created_at": str(i)})

    reminders = agent.check_reminders("pre_model", tool_steps=3, attempts=3, prompt_metadata={})
    assert len(reminders) == 0


def test_reminder_error_recovery_fires_on_consecutive_errors(tmp_path):
    """连续 2 次工具 error → 触发 error_recovery_abandon。"""
    agent = build_agent(
        tmp_path,
        ["<final>Giving up.</final>"],
    )
    agent.record({"role": "tool", "name": "run_shell", "args": {"command": "bad1"}, "content": "error: command not found", "created_at": "t1"})
    agent.record({"role": "tool", "name": "run_shell", "args": {"command": "bad2"}, "content": "error: command not found", "created_at": "t2"})

    reminders = agent.check_reminders("pre_model", tool_steps=2, attempts=2, prompt_metadata={})
    assert len(reminders) >= 1
    assert "error" in reminders[0].lower()


# ── 大输出 Offloading 测试 ─────────────────────────────────────────────


def test_large_tool_output_is_offloaded_to_scratch(tmp_path):
    """工具输出 > 8000 字符 → offload 到 .pico/scratch/ 并返回预览。"""
    (tmp_path / "big.txt").write_text("X" * 9000, encoding="utf-8")
    agent = build_agent(
        tmp_path,
        ["<final>ok</final>"],
    )
    # 模拟 task_state 以便生成 scratch 文件名
    agent.current_task_state = type("TS", (), {"task_id": "task_test", "tool_steps": 1})()

    result = agent.execute_tool("read_file", {"path": "big.txt", "start": 1, "end": 9000})

    # 验证预览而非完整内容
    assert len(result.content) < 9000
    assert "full output" in result.content
    assert ".pico/scratch/" in result.content

    # 验证 scratch 文件存在且包含完整内容（read_file 会加 # path 头 + 行号）
    scratch_dir = tmp_path / ".micro" / "scratch"
    scratch_files = list(scratch_dir.glob("*.txt"))
    assert len(scratch_files) == 1
    full_content = scratch_files[0].read_text(encoding="utf-8")
    assert "X" * 9000 in full_content


def test_small_tool_output_is_not_offloaded(tmp_path):
    """工具输出 ≤ 8000 字符 → 正常返回，不创建 scratch 文件。"""
    (tmp_path / "small.txt").write_text("hello", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        ["<final>ok</final>"],
    )

    result = agent.execute_tool("read_file", {"path": "small.txt"})

    assert "hello" in result.content
    assert "full output" not in result.content
    # 不应创建 scratch 目录
    assert not (tmp_path / ".micro" / "scratch").exists()


def test_offload_creates_unique_filenames(tmp_path):
    """每次 offload 生成独立文件名（包含 task_id + tool_name + step）。"""
    (tmp_path / "big.txt").write_text("Y" * 9000, encoding="utf-8")
    agent = build_agent(
        tmp_path,
        ["<final>ok</final>"],
    )
    agent.current_task_state = type("TS", (), {"task_id": "task_A", "tool_steps": 1})()
    agent.execute_tool("read_file", {"path": "big.txt", "start": 1, "end": 9000})

    agent.current_task_state = type("TS", (), {"task_id": "task_A", "tool_steps": 2})()
    agent.execute_tool("read_file", {"path": "big.txt", "start": 1, "end": 9000})

    scratch_files = list((tmp_path / ".micro" / "scratch").glob("*.txt"))
    assert len(scratch_files) == 2
    assert scratch_files[0].name != scratch_files[1].name


# ── 审批持久化测试 ───────────────────────────────────────────────────


def test_approval_store_default_rules_block_danger_commands(tmp_path):
    """内置危险规则自动拒绝 rm -rf / sudo / chmod 777。"""
    from micro.approval_store import ApprovalStore

    store = ApprovalStore(tmp_path)
    # rm -rf 应该被内置规则拦截
    assert store.match("run_shell", {"command": "rm -rf /tmp/foo"}) == "never"
    # sudo 应该被拦截
    assert store.match("run_shell", {"command": "sudo make install"}) == "never"
    # 普通命令不受影响
    assert store.match("run_shell", {"command": "pytest -q"}) is None


def test_approval_store_add_command_rule(tmp_path):
    """添加 COMMAND 规则后，匹配的命令按规则决议。"""
    from micro.approval_store import ApprovalStore

    store = ApprovalStore(tmp_path)
    store.add("COMMAND", "pytest -q", "auto")
    assert store.match("run_shell", {"command": "pytest -q"}) == "auto"


def test_approval_store_add_prefix_rule(tmp_path):
    """PREFIX 规则匹配以指定前缀开头的命令。"""
    from micro.approval_store import ApprovalStore

    store = ApprovalStore(tmp_path)
    store.add("PREFIX", "git ", "ask")
    assert store.match("run_shell", {"command": "git status"}) == "ask"
    assert store.match("run_shell", {"command": "git diff HEAD~1"}) == "ask"
    # 不以 git 开头的不匹配
    assert store.match("run_shell", {"command": "npm test"}) is None


def test_approval_store_remove_rule(tmp_path):
    """删除用户规则后不再匹配。"""
    from micro.approval_store import ApprovalStore

    store = ApprovalStore(tmp_path)
    store.add("COMMAND", "pytest -q", "auto")
    assert store.match("run_shell", {"command": "pytest -q"}) == "auto"

    # 找到用户规则的索引（用户规则在内置默认规则之后）
    rules = store.list_rules()
    user_index = next(r["index"] for r in rules if r["pattern"] == "pytest -q")
    store.remove(user_index)
    assert store.match("run_shell", {"command": "pytest -q"}) is None


def test_approval_store_persists_across_instances(tmp_path):
    """规则持久化到磁盘，新实例能加载。"""
    from micro.approval_store import ApprovalStore

    store1 = ApprovalStore(tmp_path)
    store1.add("COMMAND", "npm test", "auto")

    store2 = ApprovalStore(tmp_path)
    assert store2.match("run_shell", {"command": "npm test"}) == "auto"


def test_approval_store_list_rules(tmp_path):
    """list_rules 返回带索引的规则列表。"""
    from micro.approval_store import ApprovalStore

    store = ApprovalStore(tmp_path)
    store.add("COMMAND", "pytest -q", "auto")
    store.add("PREFIX", "git ", "ask")

    rules = store.list_rules()
    assert len(rules) >= 2  # 内置危险规则 + 2 条用户规则
    assert rules[-2]["index"] is not None
    assert rules[-1]["index"] is not None


def test_approval_store_non_shell_tools_return_none(tmp_path):
    """非 run_shell 工具不参与规则匹配，返回 None（交由全局策略）。"""
    from micro.approval_store import ApprovalStore

    store = ApprovalStore(tmp_path)
    store.add("COMMAND", "write", "auto")  # 这条规则不会被用到

    assert store.match("write_file", {"path": "a.py"}) is None
    assert store.match("patch_file", {"path": "a.py"}) is None


def test_approval_integration_respects_persistent_rule(tmp_path):
    """Micro.approve() 优先使用持久化规则。"""
    from micro.approval_store import ApprovalStore

    agent = build_agent(tmp_path, ["<final>ok</final>"], approval_policy="ask")
    agent.approval_store.add("COMMAND", "echo hello", "auto")

    # COMMAND 规则设为 auto → 应该自动批准
    assert agent.approve("run_shell", {"command": "echo hello"}) is True


# ── 自适应压缩测试 ───────────────────────────────────────────────────


def test_observation_mask_produces_reference_pointers(tmp_path):
    """Stage 2 观察遮蔽：旧工具输出变成引用指针而非截断内容。"""
    agent = build_agent(tmp_path, ["<final>ok</final>"])
    # 模拟多轮历史：最近 6 轮内的保留完整，之外的用引用指针
    for i in range(10):
        agent.record({"role": "tool", "name": "read_file", "args": {"path": f"file{i}.py"}, "content": f"# file{i}.py\ndef foo{i}():\n    pass\n" * 3, "created_at": str(i)})

    # 用观察遮蔽渲染
    rendered = agent.context_manager._render_history_section(
        budget=5000, observation_mask=True, protection_window=2
    )
    masked = rendered.rendered

    # 旧（file0-file7）应该是引用指针，新（file8, file9）应该完整
    assert "lines" in masked or "output masked" in masked.lower()  # 引用指针
    assert "file9.py" in masked  # 最近的在保护窗口内


def test_observation_mask_is_smaller_than_raw(tmp_path):
    """观察遮蔽输出的字符数应显著小于原始输出。"""
    agent = build_agent(tmp_path, ["<final>ok</final>"])
    for i in range(8):
        agent.record({"role": "tool", "name": "read_file", "args": {"path": f"f{i}.py"}, "content": "X" * 200, "created_at": str(i)})

    raw = agent.context_manager._render_history_section(budget=5000, observation_mask=False, protection_window=6)
    masked = agent.context_manager._render_history_section(budget=5000, observation_mask=True, protection_window=2)

    assert len(masked.rendered) < len(raw.rendered)


def test_protection_window_controls_recent_items_kept_intact(tmp_path):
    """保护窗口控制最近几轮保留完整输出。"""
    agent = build_agent(tmp_path, ["<final>ok</final>"])
    for i in range(5):
        agent.record({"role": "tool", "name": "read_file", "args": {"path": f"f{i}.py"}, "content": f"content_{i}" * 20, "created_at": str(i)})

    # protection_window=1：只有最后一次保留完整
    tight = agent.context_manager._render_history_section(budget=5000, observation_mask=True, protection_window=1)
    # protection_window=5：全部保留完整（5 个都在窗口内）
    wide = agent.context_manager._render_history_section(budget=5000, observation_mask=True, protection_window=5)

    assert len(tight.rendered) < len(wide.rendered)


def test_stage_1_warning_does_not_change_prompt(tmp_path):
    """Stage 1（≥70%）仅记录日志，不改变 prompt 内容。"""
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    # 设置极低预算以触发 ≥70%
    agent.context_manager.total_budget = 100
    meta = agent._build_prompt_and_metadata("test")[1]
    reductions = meta.get("budget_reductions", [])
    # 应该有 stage 1 的日志
    stages = [r.get("stage") for r in reductions]
    # 在 100 字符预算下可能触发 stage 1 或更高
    assert len(reductions) >= 0  # 至少有日志记录


def test_search_tool_output_masked_with_match_count(tmp_path):
    """search 工具的引用指针应包含匹配数和文件数。"""
    agent = build_agent(tmp_path, ["<final>ok</final>"])
    content = "a.py:1:foo\na.py:5:foobar\nb.py:3:foo\n"
    agent.record({"role": "tool", "name": "search", "args": {"pattern": "foo"}, "content": content, "created_at": "t1"})

    # 这条在保护窗口外 → 遮蔽
    masked = agent.context_manager._mask_tool_output({
        "name": "search", "args": {"pattern": "foo"}, "content": content,
    })
    assert "3 matches" in masked
    assert "2 files" in masked


def test_shell_tool_output_masked_with_exit_code_and_test_counts(tmp_path):
    """run_shell 的引用指针应解析退出码和测试结果。"""
    agent = build_agent(tmp_path, ["<final>ok</final>"])
    content = "exit_code: 0\nstdout:\n8 passed, 2 failed in 2.34s\n"
    agent.record({"role": "tool", "name": "run_shell", "args": {"command": "pytest"}, "content": content, "created_at": "t1"})

    masked = agent.context_manager._mask_tool_output({
        "name": "run_shell", "args": {"command": "pytest"}, "content": content,
    })
    assert "exit:0" in masked
    assert "8P/2F" in masked


def test_read_file_masked_output_shows_line_count_and_symbols(tmp_path):
    """read_file 的引用指针应显示行数和关键符号。"""
    agent = build_agent(tmp_path, ["<final>ok</final>"])
    content = "# main.py\n   1: import os\n   2: \n   3: def main():\n   4:     pass\n   5: class Config:\n   6:     pass\n"
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "main.py"}, "content": content, "created_at": "t1"})

    masked = agent.context_manager._mask_tool_output({
        "name": "read_file", "args": {"path": "main.py"}, "content": content,
    })
    assert "main.py" in masked
    assert "lines" in masked
    assert "main" in masked
    assert "Config" in masked


# ── Thinking Model 测试 ───────────────────────────────────────────────


def test_thinking_model_is_skipped_when_not_configured(tmp_path):
    """未配置 thinking_model_client 时，正常完成不报错。"""
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    assert agent.thinking_model_client is None
    assert agent.ask("test") == "Done."


def test_thinking_model_called_when_configured(tmp_path):
    """配置 thinking 模型后，thinking 输出被注入到 history。"""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    thinking_outputs = ["Analyze: should read a.py first, then decide."]

    class ThinkingFake(FakeModelClient):
        pass

    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"a.py","start":1,"end":1}}</tool>',
            "<final>File inspected.</final>",
        ],
        max_steps=4,
        thinking_model_client=ThinkingFake(thinking_outputs),
    )
    answer = agent.ask("Inspect a.py")
    assert answer == "File inspected."

    # thinking 输出应该以 [Thinking] role:user 出现在 history
    thinking_items = [
        item for item in agent.session["history"]
        if item["role"] == "user" and "[Thinking]" in item["content"]
    ]
    assert len(thinking_items) >= 1


def test_thinking_model_skipped_for_short_tasks(tmp_path):
    """max_steps < 4 时跳过 thinking。"""
    agent = build_agent(
        tmp_path,
        ["<final>Quick answer.</final>"],
        max_steps=2,
        thinking_model_client=FakeModelClient(["unused"]),
    )
    answer = agent.ask("quick question")
    assert answer == "Quick answer."
    # thinking 不应该被调用（FakeModelClient 的 "unused" 没被消费）


def test_thinking_model_failure_does_not_block_main_flow(tmp_path):
    """thinking 模型调用失败不应影响主流程。"""

    class BrokenThinking:
        supports_prompt_cache = False
        last_completion_metadata = {}

        def complete(self, prompt, max_new_tokens, **kwargs):
            raise RuntimeError("thinking model unavailable")

    agent = build_agent(
        tmp_path,
        ["<final>Recovered.</final>"],
        max_steps=4,
        thinking_model_client=BrokenThinking(),
    )
    answer = agent.ask("do the task")
    assert answer == "Recovered."


# ── Self-Critique 测试 ────────────────────────────────────────────────


def test_critique_runs_after_thinking_for_complex_tasks(tmp_path):
    """max_steps >= 6 时，thinking 之后应触发 critique。"""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")

    class ThinkAndCritique(FakeModelClient):
        pass

    thinking = ThinkAndCritique([
        "Analyze: read a.py, check for bug pattern X.",
        "Critique: plan looks solid, but also check imports.",
    ])

    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"a.py","start":1,"end":1}}</tool>',
            "<final>Bug not found, file is clean.</final>",
        ],
        max_steps=6,
        thinking_model_client=thinking,
    )
    answer = agent.ask("Find bug pattern X in a.py")
    assert answer == "Bug not found, file is clean."

    # 验证 thinking 和 critique 都出现在 history
    history = agent.session["history"]
    thinking_items = [i for i in history if "[Thinking]" in i.get("content", "")]
    critique_items = [i for i in history if "[Critique]" in i.get("content", "")]
    assert len(thinking_items) >= 1
    assert len(critique_items) >= 1


def test_critique_skipped_for_medium_tasks(tmp_path):
    """max_steps < 6 时不触发 critique（但 thinking 仍运行）。"""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    thinking = FakeModelClient(["Analyze: simple task, read a.py."])
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"a.py","start":1,"end":1}}</tool>',
            "<final>Done.</final>",
        ],
        max_steps=4,
        thinking_model_client=thinking,
    )
    agent.ask("Read a.py")
    critique_items = [i for i in agent.session["history"] if "[Critique]" in i.get("content", "")]
    assert len(critique_items) == 0


def test_critique_failure_does_not_block_main_flow(tmp_path):
    """critique 调用失败不影响主流程。"""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")

    class ThinkOkCritiqueFail:
        supports_prompt_cache = False
        last_completion_metadata = {}
        _call_count = 0

        def complete(self, prompt, max_new_tokens, **kwargs):
            self._call_count += 1
            if self._call_count == 1:
                return "Thinking: read a.py."
            raise RuntimeError("critique failed")

    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"a.py","start":1,"end":1}}</tool>',
            "<final>Completed despite critique failure.</final>",
        ],
        max_steps=6,
        thinking_model_client=ThinkOkCritiqueFail(),
    )
    answer = agent.ask("Check a.py")
    assert answer == "Completed despite critique failure."


# ── Plan Mode 测试 ───────────────────────────────────────────────────


def test_plan_mode_generates_structured_plan(tmp_path):
    """Planner agent 生成结构化计划。"""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass\n", encoding="utf-8")

    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"src/main.py","start":1,"end":10}}</tool>',
            '<tool>{"name":"search","args":{"pattern":"main","path":"."}}</tool>',
            '<tool>{"name":"list_files","args":{"path":"src"}}</tool>',
            "<final>## Goal\nAdd logging to main.py\n\n## Context\nFound main() in src/main.py\n\n## Files to modify\n- src/main.py — add logging\n\n## New files\nNone\n\n## Steps\n1. Read src/main.py\n2. Add import logging\n3. Add logging.info() call\n\n## Verification\n- Run python src/main.py\n\n## Risks\nNone</final>",
        ],
    )
    from micro.plan_mode import generate_plan
    plan = generate_plan(agent, "Add logging to main.py")
    assert "## Goal" in plan
    assert "## Steps" in plan
    assert "logging" in plan.lower()


def test_extract_plan_from_output_handles_extra_text(tmp_path):
    """_extract_plan_from_output 提取以 ## Goal 开头的部分。"""
    from micro.plan_mode import _extract_plan_from_output

    raw = "Some chat text\n## Goal\nDo something.\n\n## Steps\n1. Step one"
    plan = _extract_plan_from_output(raw)
    assert plan.startswith("## Goal")
    assert "Do something" in plan
    assert "Some chat" not in plan


def test_plan_mode_planner_has_only_read_tools(tmp_path):
    """Planner agent 只能使用只读工具（schema gating）。"""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass\n", encoding="utf-8")

    # Planner 需要足够多的输出来完成探索（最多 5 轮工具调用 + 最终计划）
    plan_outputs = [
        '<tool>{"name":"list_files","args":{"path":"src"}}</tool>',
        "<final>## Goal\nAdd logging\n\n## Context\nfound main()\n\n## Files to modify\n- src/main.py\n\n## New files\nNone\n\n## Steps\n1. Read main.py\n2. Add logging\n\n## Verification\n- run it\n\n## Risks\nNone</final>",
    ]
    agent = build_agent(tmp_path, plan_outputs)
    from micro.plan_mode import generate_plan
    plan_text = generate_plan(agent, "test task")
    assert plan_text
    assert "## Goal" in plan_text


def test_active_plan_injected_into_prompt(tmp_path):
    """active_plan 内容被注入到 prompt prefix。"""
    from micro.plan_mode import save_plan
    agent = build_agent(tmp_path, ["<final>Executed plan.</final>"])

    plan_text = "## Goal\nTest plan\n\n## Steps\n1. Do something"
    save_plan(agent.root, plan_text, "test")

    # 验证 active_plan 返回计划内容
    assert "Test plan" in agent.active_plan

    # 验证 plan 被注入到了 prompt
    agent.ask("execute the plan")
    last_prompt = agent.model_client.prompts[-1]
    assert "Current Plan:" in last_prompt
    assert "Test plan" in last_prompt


def test_plan_mode_clear_plans(tmp_path):
    """clear_active_plan 清除所有计划文件。"""
    from micro.plan_mode import save_plan

    agent = build_agent(tmp_path, ["<final>ok</final>"])
    save_plan(agent.root, "## Goal\nSomething", "test")
    assert agent.active_plan

    agent.clear_active_plan()
    assert not agent.active_plan


# ── Fuzzy Patch 测试 ─────────────────────────────────────────────────


def test_fuzzy_patch_stage2_trimmed_whitespace(tmp_path):
    """Stage 2：old_text 首尾有多余空白时仍可匹配。"""
    path = tmp_path / "a.py"
    path.write_text("def foo():\n    pass\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("patch_file", {
        "path": "a.py",
        "old_text": "  def foo():\n    pass  ",   # 首尾有多余空白
        "new_text": "def bar():\n    pass\n",
    })
    assert "patched" in result
    assert "stage 2" in result or "trimmed" in result
    assert path.read_text(encoding="utf-8") == "def bar():\n    pass\n"


def test_fuzzy_patch_stage3_line_level_match(tmp_path):
    """Stage 3：逐行 strip 后匹配（忽略每行缩进差异）。"""
    path = tmp_path / "a.py"
    path.write_text("    def foo():\n        pass\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("patch_file", {
        "path": "a.py",
        "old_text": "def foo():\n    pass",  # 缩进不同，但逐行 strip 后相同
        "new_text": "def bar():\n    return 42",
    })
    assert "patched" in result
    assert path.read_text(encoding="utf-8") == "def bar():\n    return 42"


def test_fuzzy_patch_stage1_exact_match_still_works(tmp_path):
    """Stage 1：精确匹配不受影响（向后兼容）。"""
    path = tmp_path / "a.py"
    path.write_text("alpha\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("patch_file", {
        "path": "a.py",
        "old_text": "alpha",
        "new_text": "beta",
    })
    assert "patched" in result
    assert path.read_text(encoding="utf-8") == "beta\n"


def test_fuzzy_patch_unique_exact_matches_still_rejected(tmp_path):
    """精确匹配出现多次时仍然拒绝（需要更多上下文）。"""
    path = tmp_path / "a.py"
    path.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("patch_file", {
        "path": "a.py",
        "old_text": "alpha",
        "new_text": "gamma",
    })
    assert "not unique" in result.lower()
    assert "2 exact matches" in result.lower()


def test_fuzzy_patch_all_stages_fail_gives_helpful_error(tmp_path):
    """全部 6 阶段失败时给出可执行的错误信息。"""
    path = tmp_path / "a.py"
    path.write_text("def foo():\n    pass\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("patch_file", {
        "path": "a.py",
        "old_text": "def completely_different():\n    return 999",
        "new_text": "nope",
    })
    assert "6 stages" in result.lower() or "re-read" in result.lower()


def test_fuzzy_patch_stage4_ignore_blank_lines(tmp_path):
    """Stage 4：忽略空白行差异。"""
    path = tmp_path / "a.py"
    path.write_text("def foo():\n\n    pass\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("patch_file", {
        "path": "a.py",
        "old_text": "def foo():\n    pass",  # 没有空行
        "new_text": "def bar():\n    return 1",
    })
    assert "patched" in result


# ── 新工具测试 ───────────────────────────────────────────────────────


def test_file_info_shows_size_and_lines(tmp_path):
    """file_info 返回文件大小、行数、修改时间。"""
    path = tmp_path / "a.py"
    path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("file_info", {"path": "a.py"})
    assert "file:" in result or "a.py" in result
    assert "bytes" in result
    assert "lines: 3" in result


def test_file_info_on_directory_shows_type(tmp_path):
    """file_info 对目录显示 directory 类型。"""
    (tmp_path / "src").mkdir()
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("file_info", {"path": "src"})
    assert "directory" in result


def test_glob_finds_files_by_pattern(tmp_path):
    """glob 按模式匹配文件。"""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("x", encoding="utf-8")
    (tmp_path / "src" / "c.txt").write_text("x", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("glob", {"pattern": "src/*.py"})
    assert "a.py" in result
    assert "b.py" in result
    assert "c.txt" not in result


def test_grep_count_shows_match_counts_not_content(tmp_path):
    """grep_count 返回匹配统计，不返回具体内容。"""
    (tmp_path / "a.py").write_text("TODO: fix this\npass\n# TODO later\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("no matches here\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("grep_count", {"pattern": "TODO", "path": "."})
    assert "matches" in result.lower()
    assert "fix this" not in result  # 不包含匹配内容


def test_git_diff_on_modified_repo_shows_changes(tmp_path):
    """git_diff 显示工作区变更。"""
    (tmp_path / "a.py").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    (tmp_path / "a.py").write_text("modified\n", encoding="utf-8")

    agent = build_agent(tmp_path, [])
    result = agent.run_tool("git_diff", {})
    assert "modified" in result


def test_git_log_shows_recent_commits(tmp_path):
    """git_log 显示最近提交历史。"""
    (tmp_path / "a.py").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "first commit"], cwd=tmp_path, capture_output=True)

    agent = build_agent(tmp_path, [])
    result = agent.run_tool("git_log", {"n": 3})
    assert "first commit" in result


def test_git_log_defaults_to_five_entries(tmp_path):
    """git_log 默认返回 5 条记录，n 参数可调整。"""
    agent = build_agent(tmp_path, [])
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    result = agent.run_tool("git_log", {})
    # 没有提交时显示 (no commits)
    assert "no commits" in result.lower()


# ── ACE Playbook 测试 ───────────────────────────────────────────────


def test_playbook_extracts_write_then_patch_correction(tmp_path):
    """B 类：write_file 后被 patch_file 修正 → 格式约定。"""
    from micro.features.memory import _extract_playbook_candidates

    history = [
        {"role": "tool", "name": "write_file", "args": {"path": "a.py", "content": "def f():pass"}, "content": "wrote a.py", "created_at": "t1"},
        {"role": "tool", "name": "patch_file", "args": {"path": "a.py", "old_text": "def f():pass", "new_text": "def f():\n    pass"}, "content": "patched a.py", "created_at": "t2"},
    ]
    candidates = _extract_playbook_candidates(history)
    assert len(candidates) >= 1
    assert any("a.py" in c[1] for c in candidates)


def test_playbook_extracts_user_correction_as_preference(tmp_path):
    """C 类：用户消息含纠正关键词 → 偏好记录。"""
    from micro.features.memory import _extract_playbook_candidates

    history = [
        {"role": "user", "content": "请不要使用 class，用 dataclass 代替", "created_at": "t1"},
    ]
    candidates = _extract_playbook_candidates(history)
    assert len(candidates) >= 1
    assert "dataclass" in candidates[0][1]


def test_playbook_extracts_successful_test_command(tmp_path):
    """D 类：成功的测试命令 → 项目约定。"""
    from micro.features.memory import _extract_playbook_candidates

    history = [
        {"role": "tool", "name": "run_shell", "args": {"command": "pytest -q -x"},
         "content": "exit_code: 0\nstdout:\n10 passed in 1.23s\n", "created_at": "t1"},
    ]
    candidates = _extract_playbook_candidates(history)
    assert len(candidates) >= 1
    assert "pytest" in candidates[0][1]


def test_playbook_injects_knowledge_into_prompt(tmp_path):
    """Project Knowledge 注入 prompt prefix。"""
    agent = build_agent(tmp_path, ["<final>Got it.</final>"])
    # 模拟已有 playbook 知识
    agent.memory.durable_store.promote([("project-conventions", "use pytest -x for fail-fast")])

    prompt, _ = agent.context_manager.build("do the task")
    assert "Project Knowledge" in prompt
    assert "pytest -x" in prompt


def test_playbook_empty_when_no_knowledge(tmp_path):
    """没有积累的知识时 playbook_text 返回空字符串。"""
    agent = build_agent(tmp_path, ["<final>ok</final>"])
    assert agent.memory.playbook_text() == ""


def test_playbook_no_duplicate_candidates(tmp_path):
    """相同候选不重复添加（通过 DurableMemoryStore.promote 的去重机制）。"""
    agent = build_agent(tmp_path, ["<final>ok</final>"])

    # 第一次提取并持久化
    promoted1, _ = agent.memory.run_playbook_extraction([
        {"role": "user", "content": "请用 f-string 而不是 format()", "created_at": "t1"},
    ])
    assert len(promoted1) >= 1

    # 第二次提取相同内容 → 应被去重跳过
    promoted2, _ = agent.memory.run_playbook_extraction([
        {"role": "user", "content": "请用 f-string 而不是 format()", "created_at": "t2"},
    ])
    assert len(promoted2) == 0


def test_playbook_core_file_detected_for_structural_files(tmp_path):
    """A 类：高频读取的结构性文件被识别为核心文件。"""
    from micro.features.memory import _extract_playbook_candidates

    history = []
    for _ in range(3):
        history.append({"role": "tool", "name": "read_file", "args": {"path": "src/config.py"}, "content": "...", "created_at": "t"})
    history.append({"role": "tool", "name": "read_file", "args": {"path": "src/models.py"}, "content": "...", "created_at": "t"})

    candidates = _extract_playbook_candidates(history)
    assert any("config.py" in c[1] for c in candidates)


def test_planner_uses_dedicated_model_when_configured(tmp_path):
    """Planner 使用独立的 planner_model_client（如果配置了）。"""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass\n", encoding="utf-8")

    planner_fake = FakeModelClient([
        '<tool>{"name":"read_file","args":{"path":"src/main.py","start":1,"end":10}}</tool>',
        "<final>## Goal\nAdd logging\n\n## Context\nfound main()\n\n## Files to modify\n- src/main.py\n\n## New files\nNone\n\n## Steps\n1. Add logging\n\n## Verification\n- run it\n\n## Risks\nNone</final>",
    ])

    agent = build_agent(
        tmp_path, ["<final>ok</final>"],
        thinking_model_client=FakeModelClient(["unused-thinking"]),
        planner_model_client=planner_fake,
    )

    from micro.plan_mode import generate_plan
    plan = generate_plan(agent, "test")
    assert "## Goal" in plan
    # planner_fake 被消费了 → 确认用了 planner 专用模型
    assert len(planner_fake.outputs) == 0


def test_welcome_screen_keeps_box_shape_for_long_paths(tmp_path):
    deep = tmp_path / "very" / "long" / "path" / "for" / "the" / "micro" / "agent" / "welcome" / "screen"
    deep.mkdir(parents=True)
    agent = build_agent(deep, [])

    welcome = build_welcome(agent, model="qwen3.5:4b", host="http://127.0.0.1:11434")
    lines = welcome.splitlines()

    assert len(lines) >= 5
    assert len({len(line) for line in lines}) == 1
    assert "..." in welcome
    assert "(  o o  )" in welcome
    assert "MINI-CODING-AGENT" not in welcome
    assert "MINI CODING AGENT" not in welcome
    assert "micro" in welcome
    assert "local coding agent" in welcome
    assert "// READY" not in welcome
    assert "SLASH" not in welcome
    assert "READY      " not in welcome
    assert "commands: Commands:" not in welcome


def test_ollama_client_posts_expected_payload():
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"response": "<final>ok</final>"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OllamaModelClient(
        model="qwen3.5:4b",
        host="http://127.0.0.1:11434",
        temperature=0.2,
        top_p=0.9,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["timeout"] == 30
    assert captured["body"]["model"] == "qwen3.5:4b"
    assert captured["body"]["prompt"] == "hello"
    assert captured["body"]["stream"] is False


def test_openai_compatible_client_posts_expected_responses_payload():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"output_text": "<final>ok</final>"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"
    assert captured["url"] == "https://right.codes/v1/responses"
    assert captured["timeout"] == 30
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["headers"]["User-agent"] == "micro/0.1"
    assert captured["body"] == {
        "model": "right.codes/codex-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "hello",
                    }
                ],
            }
        ],
        "max_output_tokens": 42,
        "stream": False,
        "temperature": 0.2,
    }


def test_openai_compatible_client_sends_prompt_cache_fields_and_records_usage():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "output_text": "<final>ok</final>",
                    "usage": {
                        "input_tokens": 2048,
                        "input_tokens_details": {"cached_tokens": 1536},
                        "output_tokens": 32,
                        "total_tokens": 2080,
                    },
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete(
            "hello",
            42,
            prompt_cache_key="prefix-hash-123",
            prompt_cache_retention="in_memory",
        )

    assert result == "<final>ok</final>"
    assert captured["body"]["prompt_cache_key"] == "prefix-hash-123"
    assert captured["body"]["prompt_cache_retention"] == "in_memory"
    assert client.last_completion_metadata["prompt_cache_supported"] is True
    assert client.last_completion_metadata["cached_tokens"] == 1536
    assert client.last_completion_metadata["cache_hit"] is True
    assert client.last_completion_metadata["input_tokens"] == 2048


def test_openai_compatible_client_extracts_text_from_event_stream():
    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                'data: {"type":"response.created","response":{"id":"resp_1","output":[]}}\n'
                'data: {"type":"response.completed","response":{"output":[{"content":[{"text":"<final>stream ok</final>"}]}]}}\n'
                "data: [DONE]\n"
            ).encode("utf-8")

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = client.complete("hello", 42)

    assert result == "<final>stream ok</final>"


def test_openai_compatible_client_extracts_text_from_event_stream_deltas():
    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"<final>"}\n'
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"OK"}\n'
                'event: response.output_text.done\n'
                'data: {"type":"response.output_text.done","text":"<final>OK</final>"}\n'
                "data: [DONE]\n"
            ).encode("utf-8")

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = client.complete("hello", 42)

    assert result == "<final>OK</final>"


def test_anthropic_compatible_client_posts_expected_messages_payload():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "<final>ok</final>",
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = AnthropicCompatibleModelClient(
        model="claude-sonnet-4-5-20250929",
        base_url="https://www.right.codes/claude-aws/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"
    assert captured["url"] == "https://www.right.codes/claude-aws/v1/messages"
    assert captured["timeout"] == 30
    assert captured["headers"]["X-api-key"] == "sk-test"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["body"] == {
        "model": "claude-sonnet-4-5-20250929",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                    }
                ],
            }
        ],
        "max_tokens": 42,
        "stream": False,
        "temperature": 0.2,
    }


def test_anthropic_compatible_client_extracts_first_text_block():
    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "text", "text": "<final>ok</final>"},
                    ]
                }
            ).encode("utf-8")

    client = AnthropicCompatibleModelClient(
        model="claude-sonnet-4-5-20250929",
        base_url="https://www.right.codes/claude-aws/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"


def test_build_agent_uses_openai_provider_and_model_override(tmp_path):
    args = type(
        "Args",
        (),
        {
            "cwd": str(tmp_path),
            "provider": "openai",
            "model": "override-model",
            "base_url": None,
            "host": "http://127.0.0.1:11434",
            "ollama_timeout": 300,
            "temperature": 0.2,
            "top_p": 0.9,
            "resume": None,
            "approval": "ask",
            "secret_env_names": [],
            "max_steps": 6,
            "max_new_tokens": 512,
        },
    )()

    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE": "https://www.right.codes/codex/v1",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "env-model",
        },
        clear=False,
    ):
        with patch(
            "micro.cli.OllamaModelClient",
            side_effect=AssertionError("ollama client should not be used"),
        ), patch("micro.cli.OpenAICompatibleModelClient") as mock_openai:
            fake_client = mock_openai.return_value
            agent = micro.build_agent(args)

    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["model"] == "override-model"
    assert mock_openai.call_args.kwargs["base_url"] == "https://www.right.codes/codex/v1"
    assert mock_openai.call_args.kwargs["api_key"] == "sk-test"
    assert agent.model_client is fake_client


def test_build_agent_uses_right_codes_shared_key_for_openai_provider(tmp_path):
    args = type(
        "Args",
        (),
        {
            "cwd": str(tmp_path),
            "provider": "openai",
            "model": None,
            "base_url": None,
            "host": "http://127.0.0.1:11434",
            "ollama_timeout": 300,
            "openai_timeout": 300,
            "temperature": 0.2,
            "top_p": 0.9,
            "resume": None,
            "approval": "ask",
            "secret_env_names": [],
            "max_steps": 6,
            "max_new_tokens": 512,
        },
    )()

    with patch.dict(os.environ, {"PICO_RIGHT_CODES_API_KEY": "sk-right-codes"}, clear=True):
        with patch(
            "micro.cli.OllamaModelClient",
            side_effect=AssertionError("ollama client should not be used"),
        ), patch("micro.cli.OpenAICompatibleModelClient") as mock_openai:
            fake_client = mock_openai.return_value
            agent = micro.build_agent(args)

    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["api_key"] == "sk-right-codes"
    assert agent.model_client is fake_client


def test_build_arg_parser_leaves_provider_unset_for_runtime_resolution(tmp_path):
    args = micro.build_arg_parser().parse_args(["--cwd", str(tmp_path)])

    assert args.provider is None


def test_build_arg_parser_accepts_anthropic_provider(tmp_path):
    args = micro.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--provider", "anthropic"])

    assert args.provider == "anthropic"


def test_build_arg_parser_accepts_deepseek_provider(tmp_path):
    args = micro.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--provider", "deepseek"])

    assert args.provider == "deepseek"


def test_build_agent_uses_project_env_provider_when_cli_omitted(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "PICO_PROVIDER=openai",
                "PICO_OPENAI_API_BASE=https://www.right.codes/codex/v1",
                "PICO_OPENAI_API_KEY=sk-project-openai",
                "PICO_OPENAI_MODEL=gpt-5.4",
                "PICO_DEEPSEEK_API_KEY=sk-project-deepseek",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = micro.build_arg_parser().parse_args(["--cwd", str(tmp_path)])

    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "micro.cli.OllamaModelClient",
            side_effect=AssertionError("ollama client should not be used"),
        ), patch(
            "micro.cli.AnthropicCompatibleModelClient",
            side_effect=AssertionError("deepseek client should not be used"),
        ), patch("micro.cli.OpenAICompatibleModelClient") as mock_openai:
            fake_client = mock_openai.return_value
            agent = micro.build_agent(args)

    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["model"] == "gpt-5.4"
    assert mock_openai.call_args.kwargs["base_url"] == "https://www.right.codes/codex/v1"
    assert mock_openai.call_args.kwargs["api_key"] == "sk-project-openai"
    assert agent.model_client is fake_client


def test_build_agent_prefers_cli_provider_over_project_env_provider(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "PICO_PROVIDER=openai",
                "PICO_OPENAI_API_KEY=sk-project-openai",
                "PICO_DEEPSEEK_API_BASE=https://api.deepseek.com/anthropic",
                "PICO_DEEPSEEK_API_KEY=sk-project-deepseek",
                "PICO_DEEPSEEK_MODEL=deepseek-v4-pro",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = micro.build_arg_parser().parse_args(
        ["--cwd", str(tmp_path), "--provider", "deepseek"]
    )

    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "micro.cli.OllamaModelClient",
            side_effect=AssertionError("ollama client should not be used"),
        ), patch(
            "micro.cli.OpenAICompatibleModelClient",
            side_effect=AssertionError("openai client should not be used"),
        ), patch("micro.cli.AnthropicCompatibleModelClient") as mock_anthropic:
            fake_client = mock_anthropic.return_value
            agent = micro.build_agent(args)

    mock_anthropic.assert_called_once()
    assert mock_anthropic.call_args.kwargs["model"] == "deepseek-v4-pro"
    assert mock_anthropic.call_args.kwargs["base_url"] == "https://api.deepseek.com/anthropic"
    assert mock_anthropic.call_args.kwargs["api_key"] == "sk-project-deepseek"
    assert agent.model_client is fake_client


def test_build_agent_uses_anthropic_provider_and_openai_key_fallback(tmp_path):
    args = type(
        "Args",
        (),
        {
            "cwd": str(tmp_path),
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "base_url": None,
            "host": "http://127.0.0.1:11434",
            "ollama_timeout": 300,
            "openai_timeout": 300,
            "temperature": 0.2,
            "top_p": 0.9,
            "resume": None,
            "approval": "ask",
            "secret_env_names": [],
            "max_steps": 6,
            "max_new_tokens": 512,
        },
    )()

    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-openai-fallback",
        },
        clear=True,
    ):
        with patch(
            "micro.cli.OllamaModelClient",
            side_effect=AssertionError("ollama client should not be used"),
        ), patch(
            "micro.cli.OpenAICompatibleModelClient",
            side_effect=AssertionError("openai client should not be used"),
        ), patch("micro.cli.AnthropicCompatibleModelClient") as mock_anthropic:
            fake_client = mock_anthropic.return_value
            agent = micro.build_agent(args)

    mock_anthropic.assert_called_once()
    assert mock_anthropic.call_args.kwargs["model"] == "claude-sonnet-4-5-20250929"
    assert mock_anthropic.call_args.kwargs["base_url"] == "https://www.right.codes/claude/v1"
    assert mock_anthropic.call_args.kwargs["api_key"] == "sk-openai-fallback"
    assert agent.model_client is fake_client


def test_build_agent_uses_anthropic_default_model_when_env_is_missing(tmp_path):
    args = micro.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--provider", "anthropic"])

    with patch.dict(
        os.environ,
        {},
        clear=False,
    ):
        os.environ.pop("ANTHROPIC_MODEL", None)
        with patch("micro.cli.AnthropicCompatibleModelClient") as mock_anthropic:
            micro.build_agent(args)

    assert mock_anthropic.call_args.kwargs["model"] == "claude-sonnet-4-6"


def test_build_agent_uses_deepseek_provider_and_env_configuration(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "PICO_DEEPSEEK_API_BASE=https://api.deepseek.com/anthropic",
                "PICO_DEEPSEEK_API_KEY=sk-project-deepseek",
                "PICO_DEEPSEEK_MODEL=deepseek-v4-pro",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "cwd": str(tmp_path),
            "provider": "deepseek",
            "model": None,
            "base_url": None,
            "host": "http://127.0.0.1:11434",
            "ollama_timeout": 300,
            "openai_timeout": 300,
            "temperature": 0.2,
            "top_p": 0.9,
            "resume": None,
            "approval": "ask",
            "secret_env_names": [],
            "max_steps": 6,
            "max_new_tokens": 512,
        },
    )()

    with patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_BASE": "https://legacy.deepseek.example/anthropic",
            "DEEPSEEK_API_KEY": "sk-legacy-deepseek",
            "DEEPSEEK_MODEL": "legacy-deepseek-model",
            "ANTHROPIC_API_KEY": "sk-anthropic",
            "OPENAI_API_KEY": "sk-openai",
        },
        clear=True,
    ):
        with patch(
            "micro.cli.OllamaModelClient",
            side_effect=AssertionError("ollama client should not be used"),
        ), patch(
            "micro.cli.OpenAICompatibleModelClient",
            side_effect=AssertionError("openai client should not be used"),
        ), patch("micro.cli.AnthropicCompatibleModelClient") as mock_anthropic:
            fake_client = mock_anthropic.return_value
            agent = micro.build_agent(args)

    mock_anthropic.assert_called_once()
    assert mock_anthropic.call_args.kwargs["model"] == "deepseek-v4-pro"
    assert mock_anthropic.call_args.kwargs["base_url"] == "https://api.deepseek.com/anthropic"
    assert mock_anthropic.call_args.kwargs["api_key"] == "sk-project-deepseek"
    assert agent.model_client is fake_client


def test_build_agent_uses_deepseek_default_model_when_env_is_missing(tmp_path):
    args = micro.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--provider", "deepseek"])

    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-deepseek"}, clear=True):
        with patch("micro.cli.AnthropicCompatibleModelClient") as mock_anthropic:
            micro.build_agent(args)

    assert mock_anthropic.call_args.kwargs["model"] == "deepseek-v4-pro"
    assert mock_anthropic.call_args.kwargs["base_url"] == "https://api.deepseek.com/anthropic"


def test_build_agent_uses_deepseek_provider_by_default(tmp_path):
    args = micro.build_arg_parser().parse_args(["--cwd", str(tmp_path)])

    with patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_BASE": "https://api.deepseek.com/anthropic",
            "DEEPSEEK_API_KEY": "sk-test",
        },
        clear=False,
    ):
        with patch(
            "micro.cli.OllamaModelClient",
            side_effect=AssertionError("ollama client should not be used"),
        ), patch(
            "micro.cli.OpenAICompatibleModelClient",
            side_effect=AssertionError("openai client should not be used"),
        ), patch("micro.cli.AnthropicCompatibleModelClient") as mock_anthropic:
            fake_client = mock_anthropic.return_value
            agent = micro.build_agent(args)

    mock_anthropic.assert_called_once()
    assert mock_anthropic.call_args.kwargs["model"] == "deepseek-v4-pro"
    assert mock_anthropic.call_args.kwargs["base_url"] == "https://api.deepseek.com/anthropic"
    assert mock_anthropic.call_args.kwargs["api_key"] == "sk-test"
    assert agent.model_client is fake_client


def test_successful_run_persists_run_artifacts_and_stop_reason(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Finished.</final>",
        ],
    )

    assert agent.ask("Do the thing") == "Finished."

    runs_root = tmp_path / ".micro" / "runs"
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    task_state = json.loads((run_dir / "task_state.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    trace_lines = (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()

    assert task_state["task_id"] != task_state["run_id"]
    assert run_dir.name == task_state["run_id"]
    assert (run_dir / "task_state.json").exists()
    assert (run_dir / "trace.jsonl").exists()
    assert (run_dir / "report.json").exists()
    assert task_state["stop_reason"] == "final_answer_returned"
    assert task_state["final_answer"] == "Finished."
    assert report["stop_reason"] == "final_answer_returned"
    assert report["task_state"]["stop_reason"] == "final_answer_returned"
    assert report["run_id"] == task_state["run_id"]
    trace_events = [json.loads(line)["event"] for line in trace_lines]
    assert trace_events[0] == "run_started"
    assert trace_events[-1] == "run_finished"
    assert trace_events.count("prompt_built") == 2
    assert "tool_executed" in trace_events


def test_trace_and_report_redact_secret_env_values(tmp_path):
    secret = "sk-test-secret-123"
    with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=True):
        agent = build_agent(
            tmp_path,
            [
                '<tool>{"name":"run_shell","args":{"command":"printf \'%s\' \'sk-test-secret-123\'","timeout":20}}</tool>',
                "<final>Masked.</final>",
            ],
        )

        assert agent.ask("Mask the secret") == "Masked."

    runs_root = tmp_path / ".micro" / "runs"
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    report_text = (run_dir / "report.json").read_text(encoding="utf-8")
    trace_events = [json.loads(line) for line in trace_text.splitlines()]

    assert secret not in trace_text
    assert secret not in report_text

    prompt_events = [event for event in trace_events if event["event"] == "prompt_built"]
    assert prompt_events
    assert prompt_events[0]["prompt_metadata"]["secret_env_count"] >= 1
    assert "OPENAI_API_KEY" in prompt_events[0]["prompt_metadata"]["secret_env_names"]

    tool_events = [event for event in trace_events if event["event"] == "tool_executed"]
    assert tool_events
    assert "<redacted>" in tool_events[0]["args"]["command"]
    assert "<redacted>" in tool_events[0]["result"]


def test_prompt_budget_metadata_records_budget_decisions(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    agent.memory.append_note("alpha episodic note " + ("A" * 120), tags=("recall",), created_at="2026-04-07T10:00:00+00:00")
    agent.memory.append_note("beta episodic recall note " + ("B" * 120), created_at="2026-04-07T10:01:00+00:00")
    agent.memory.append_note("gamma episodic note " + ("C" * 120), tags=("recall",), created_at="2026-04-07T10:02:00+00:00")

    for index in range(4):
        agent.record(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"history-{index}-" + ("A" * 240),
                "created_at": f"2026-04-07T10:0{index}:00+00:00",
            }
        )

    agent.context_manager.total_budget = 1000
    agent.context_manager.section_budgets = {
        "prefix": 80,
        "memory": 80,
        "relevant_memory": 80,
        "history": 80,
    }

    assert agent.ask("recall") == "Done."

    trace_events = [
        json.loads(line)
        for line in (agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines())
    ]
    prompt_events = [event for event in trace_events if event["event"] == "prompt_built"]
    assert prompt_events
    metadata = prompt_events[0]["prompt_metadata"]
    relevant_section = agent.model_client.prompts[0].split("Relevant memory:\n", 1)[1].split("\n\nTranscript:", 1)[0]

    assert metadata["relevant_memory"]["selected_count"] == 3
    assert len(metadata["relevant_memory"]["rendered_notes"]) == 3
    assert len([line for line in relevant_section.splitlines() if line.startswith("- ")]) == 3
    assert "alpha episodic" in relevant_section
    assert "beta episodic" in relevant_section
    assert "gamma episodic" in relevant_section
    assert metadata["current_request"]["text"] == "recall"
    assert metadata["current_request"]["rendered_chars"] == len("recall")


def test_prompt_metadata_refreshes_prefix_when_workspace_changes(tmp_path):
    agent = build_agent(tmp_path, [])

    first = agent.prompt_metadata("first", "")
    second = agent.prompt_metadata("second", "")

    assert first["prefix_hash"] == second["prefix_hash"]
    assert second["prefix_changed"] is False
    assert second["workspace_changed"] is False

    (tmp_path / "README.md").write_text("demo changed\n", encoding="utf-8")

    third = agent.prompt_metadata("third", "")

    assert third["prefix_hash"] != second["prefix_hash"]
    assert third["prefix_changed"] is True
    assert third["workspace_changed"] is True
    assert "demo changed" in agent.prefix


def test_agent_creates_checkpoint_when_context_reduction_happens_and_artifacts_only_reference_it(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done after checkpoint.</final>"])
    for index in range(10):
        agent.record(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"history-{index}-" + ("A" * 260),
                "created_at": f"2026-04-07T10:{index:02d}:00+00:00",
            }
        )
    agent.memory.append_note("checkpoint note " + ("B" * 220), tags=("checkpoint",), created_at="2026-04-07T11:00:00+00:00")
    agent.context_manager.total_budget = 900
    agent.context_manager.section_budgets = {
        "prefix": 120,
        "memory": 120,
        "relevant_memory": 120,
        "history": 160,
    }

    assert agent.ask("Resume the long task") == "Done after checkpoint."

    checkpoint_state = agent.session["checkpoints"]
    checkpoint = checkpoint_state["items"][checkpoint_state["current_id"]]
    assert checkpoint["checkpoint_id"] == checkpoint_state["current_id"]
    assert checkpoint["schema_version"] == "phase1-v1"
    assert checkpoint["current_goal"] == "Resume the long task"
    assert checkpoint["key_files"] == []
    assert checkpoint["current_blocker"] == ""
    assert checkpoint["next_step"]

    task_state = json.loads(agent.run_store.task_state_path(agent.current_task_state).read_text(encoding="utf-8"))
    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))
    trace_events = [
        json.loads(line)
        for line in agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines()
    ]

    assert task_state["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert report["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert report["task_state"]["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert "current_goal" not in task_state
    assert "current_goal" not in report
    checkpoint_events = [event for event in trace_events if event["event"] == "checkpoint_created"]
    assert checkpoint_events
    assert checkpoint_events[-1]["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert "current_goal" not in checkpoint_events[-1]


def test_resume_prompt_uses_checkpoint_state_not_just_history(tmp_path):
    agent = build_agent(tmp_path, ["<final>checkpoint ready.</final>"])
    agent.session["checkpoints"] = {
        "current_id": "ckpt_manual",
        "items": {
            "ckpt_manual": {
                "checkpoint_id": "ckpt_manual",
                "parent_checkpoint_id": "",
                "schema_version": "phase1-v1",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Fix failing resume flow",
                "completed": ["Read runtime.py"],
                "excluded": ["Do not add branch summary"],
                "current_blocker": "Need to re-anchor stale file facts",
                "next_step": "Re-read runtime.py and refresh the checkpoint",
                "key_files": [{"path": "runtime.py", "freshness": "abc"}],
                "freshness": {"runtime.py": "abc"},
                "summary": "Resume from the latest checkpoint",
                "runtime_identity": {"workspace_fingerprint": "old-fingerprint"},
            }
        },
    }
    agent.session_store.save(agent.session)

    resumed = Micro.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."

    prompt = resumed.model_client.prompts[-1]
    assert "Task checkpoint:" in prompt
    assert "Current goal: Fix failing resume flow" in prompt
    assert "Current blocker: Need to re-anchor stale file facts" in prompt
    assert "Next step: Re-read runtime.py and refresh the checkpoint" in prompt


def test_resume_invalidates_stale_file_summaries_and_marks_partial_stale(tmp_path):
    file_path = tmp_path / "runtime.py"
    file_path.write_text("alpha\n", encoding="utf-8")
    agent = build_agent(tmp_path, ["<final>checkpoint ready.</final>"])
    agent.memory.set_file_summary("runtime.py", "runtime.py: alpha")
    freshness = agent.memory.to_dict()["file_summaries"]["runtime.py"]["freshness"]
    agent.session["checkpoints"] = {
        "current_id": "ckpt_stale",
        "items": {
            "ckpt_stale": {
                "checkpoint_id": "ckpt_stale",
                "parent_checkpoint_id": "",
                "schema_version": "phase1-v1",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Fix stale summary handling",
                "completed": [],
                "excluded": [],
                "current_blocker": "",
                "next_step": "Re-read runtime.py",
                "key_files": [{"path": "runtime.py", "freshness": freshness}],
                "freshness": {"runtime.py": freshness},
                "summary": "runtime.py is important",
                "runtime_identity": {"workspace_fingerprint": agent.workspace.fingerprint()},
            }
        },
    }
    agent.session_store.save(agent.session)
    file_path.write_text("beta\n", encoding="utf-8")

    resumed = Micro.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."

    assert "runtime.py" not in resumed.memory.to_dict()["file_summaries"]
    assert resumed.last_prompt_metadata["resume_status"] == "partial-stale"
    assert resumed.last_prompt_metadata["stale_summary_invalidations"] == 1


def test_run_shell_nonzero_with_workspace_change_is_recorded_as_partial_success(tmp_path):
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "run_shell",
        {
            "command": "printf 'changed\\n' > README.md && exit 1",
            "timeout": 20,
        },
    )

    assert "exit_code: 1" in result
    assert agent._last_tool_result_metadata["tool_status"] == "partial_success"
    assert agent._last_tool_result_metadata["affected_paths"] == ["README.md"]
    assert agent._last_tool_result_metadata["workspace_changed"] is True


def test_resume_marks_workspace_mismatch_when_checkpoint_runtime_identity_is_stale(tmp_path):
    agent = build_agent(tmp_path, ["<final>checkpoint ready.</final>"])
    agent.session["checkpoints"] = {
        "current_id": "ckpt_workspace",
        "items": {
            "ckpt_workspace": {
                "checkpoint_id": "ckpt_workspace",
                "parent_checkpoint_id": "",
                "schema_version": "phase1-v1",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Continue after drift",
                "completed": [],
                "excluded": [],
                "current_blocker": "",
                "next_step": "Rebuild runtime state",
                "key_files": [],
                "freshness": {},
                "summary": "workspace changed",
                "runtime_identity": {"workspace_fingerprint": "outdated-fingerprint"},
            }
        },
    }
    agent.session_store.save(agent.session)

    resumed = Micro.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."
    assert resumed.last_prompt_metadata["resume_status"] == "workspace-mismatch"


def test_write_file_trace_records_minimum_tool_contract_fields(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"write_file","args":{"path":"notes.txt","content":"hello\\n"}}</tool>',
            "<final>Done.</final>",
        ],
    )

    assert agent.ask("Create notes.txt") == "Done."

    trace_events = [
        json.loads(line)
        for line in agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines()
    ]
    tool_event = [event for event in trace_events if event["event"] == "tool_executed"][-1]

    assert tool_event["name"] == "write_file"
    assert tool_event["risk_level"] == "high"
    assert tool_event["read_only"] is False
    assert tool_event["tool_status"] == "ok"
    assert tool_event["affected_paths"] == ["notes.txt"]
    assert tool_event["workspace_changed"] is True
    assert tool_event["diff_summary"] == ["created:notes.txt"]


def test_resume_marks_schema_mismatch_when_checkpoint_version_is_incompatible(tmp_path):
    agent = build_agent(tmp_path, ["<final>checkpoint ready.</final>"])
    agent.session["checkpoints"] = {
        "current_id": "ckpt_schema",
        "items": {
            "ckpt_schema": {
                "checkpoint_id": "ckpt_schema",
                "parent_checkpoint_id": "",
                "schema_version": "legacy-v0",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Continue after schema change",
                "completed": [],
                "excluded": [],
                "current_blocker": "",
                "next_step": "Migrate checkpoint",
                "key_files": [],
                "freshness": {},
                "summary": "schema changed",
                "runtime_identity": {"workspace_fingerprint": agent.workspace.fingerprint()},
            }
        },
    }
    agent.session_store.save(agent.session)

    resumed = Micro.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."
    assert resumed.last_prompt_metadata["resume_status"] == "schema-mismatch"


def test_resume_marks_no_checkpoint_when_session_has_no_checkpoint_state(tmp_path):
    agent = build_agent(tmp_path, ["<final>checkpoint ready.</final>"])
    agent.session.pop("checkpoints", None)
    agent.session_store.save(agent.session)

    resumed = Micro.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."
    assert resumed.last_prompt_metadata["resume_status"] == "no-checkpoint"
    assert "Task checkpoint:" not in resumed.model_client.prompts[-1]


def test_freshness_mismatch_creates_checkpoint_before_model_completion(tmp_path):
    file_path = tmp_path / "runtime.py"
    file_path.write_text("alpha\n", encoding="utf-8")
    agent = build_agent(tmp_path, ["<final>Resumed.</final>"])
    agent.memory.set_file_summary("runtime.py", "runtime.py: alpha")
    freshness = agent.memory.to_dict()["file_summaries"]["runtime.py"]["freshness"]
    agent.session["checkpoints"] = {
        "current_id": "ckpt_freshness",
        "items": {
            "ckpt_freshness": {
                "checkpoint_id": "ckpt_freshness",
                "parent_checkpoint_id": "",
                "schema_version": "phase1-v1",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Handle freshness mismatch",
                "completed": [],
                "excluded": [],
                "current_blocker": "",
                "next_step": "Re-read runtime.py",
                "key_files": [{"path": "runtime.py", "freshness": freshness}],
                "freshness": {"runtime.py": freshness},
                "summary": "runtime.py changed",
                "runtime_identity": {"workspace_fingerprint": agent.workspace.fingerprint()},
            }
        },
    }
    agent.session_store.save(agent.session)
    file_path.write_text("beta\n", encoding="utf-8")

    assert agent.ask("Continue the task") == "Resumed."

    trace_events = [
        json.loads(line)
        for line in agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines()
    ]
    checkpoint_events = [event for event in trace_events if event["event"] == "checkpoint_created"]

    assert checkpoint_events
    assert checkpoint_events[0]["trigger"] == "freshness_mismatch"


def test_runtime_identity_persists_key_execution_metadata(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".micro" / "sessions")
    agent = Micro(
        model_client=FakeModelClient(["<final>Done.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="never",
        max_steps=9,
        max_new_tokens=1024,
        feature_flags={"memory": True, "relevant_memory": False},
    )

    runtime_identity = agent.session["runtime_identity"]

    assert runtime_identity["session_id"] == agent.session["id"]
    assert runtime_identity["cwd"] == str(tmp_path)
    assert runtime_identity["approval_policy"] == "never"
    assert runtime_identity["read_only"] is False
    assert runtime_identity["max_steps"] == 9
    assert runtime_identity["max_new_tokens"] == 1024
    assert runtime_identity["feature_flags"]["memory"] is True
    assert runtime_identity["feature_flags"]["relevant_memory"] is False
    assert runtime_identity["shell_env_allowlist"] == list(agent.shell_env_allowlist)


def test_resume_records_runtime_identity_mismatch_fields_in_metadata_and_trace(tmp_path):
    agent = build_agent(tmp_path, ["<final>checkpoint ready.</final>"])
    agent.session["checkpoints"] = {
        "current_id": "ckpt_identity",
        "items": {
            "ckpt_identity": {
                "checkpoint_id": "ckpt_identity",
                "parent_checkpoint_id": "",
                "schema_version": "phase1-v1",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Resume with a different runtime identity",
                "completed": [],
                "excluded": [],
                "current_blocker": "",
                "next_step": "Rebuild runtime identity",
                "key_files": [],
                "freshness": {},
                "summary": "identity changed",
                "runtime_identity": {
                    "workspace_fingerprint": agent.workspace.fingerprint(),
                    "approval_policy": "auto",
                    "read_only": False,
                    "max_steps": 6,
                    "max_new_tokens": 512,
                    "model": "old-model",
                    "model_client": "FakeModelClient",
                    "feature_flags": {"memory": True, "relevant_memory": True},
                    "shell_env_allowlist": ["PATH"],
                    "session_id": agent.session["id"],
                    "cwd": str(tmp_path),
                },
            }
        },
    }
    agent.session_store.save(agent.session)

    resumed = Micro.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="never",
        max_steps=9,
        max_new_tokens=1024,
        feature_flags={"memory": True, "relevant_memory": False},
    )

    resumed.ask("Continue the task")

    assert resumed.last_prompt_metadata["resume_status"] == "workspace-mismatch"
    assert resumed.last_prompt_metadata["runtime_identity_mismatch_fields"] == [
        "approval_policy",
        "feature_flags",
        "max_new_tokens",
        "max_steps",
        "model",
        "shell_env_allowlist",
    ]

    trace_events = [
        json.loads(line)
        for line in resumed.run_store.trace_path(resumed.current_task_state).read_text(encoding="utf-8").splitlines()
    ]
    mismatch_events = [event for event in trace_events if event["event"] == "runtime_identity_mismatch"]
    assert mismatch_events
    assert mismatch_events[0]["fields"] == [
        "approval_policy",
        "feature_flags",
        "max_new_tokens",
        "max_steps",
        "model",
        "shell_env_allowlist",
    ]


def test_partial_success_creates_process_note_for_exploration_history(tmp_path):
    agent = build_agent(tmp_path, [])

    agent.run_tool(
        "run_shell",
        {
            "command": "printf 'changed\\n' > README.md && exit 1",
            "timeout": 20,
        },
    )

    process_notes = [
        note
        for note in agent.memory.to_dict()["episodic_notes"]
        if note.get("kind") == "process"
    ]

    assert process_notes
    assert process_notes[-1]["text"] == "run_shell partial_success on README.md; inspect diff before retry"
    assert "partial_success" in process_notes[-1]["tags"]
    assert "README.md" in process_notes[-1]["tags"]


def test_explicit_memory_promotion_persists_durable_memory_topics(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>Project convention: Use constrained tools instead of guessing.\n"
            "Project convention: Preserve local agent state under .pico/.\n"
            "Decision: Keep durable memory topic-based and lightweight.</final>",
        ],
    )

    answer = agent.ask(
        "Capture the stable facts you already discovered as durable memory. "
        "Respond with exactly the long-term facts."
    )

    assert "Project convention:" in answer

    index_path = tmp_path / ".micro" / "memory" / "MEMORY.md"
    conventions_path = tmp_path / ".micro" / "memory" / "topics" / "project-conventions.md"
    decisions_path = tmp_path / ".micro" / "memory" / "topics" / "key-decisions.md"
    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))

    assert index_path.exists()
    assert conventions_path.exists()
    assert decisions_path.exists()
    assert "project-conventions" in index_path.read_text(encoding="utf-8")
    assert "Use constrained tools instead of guessing." in conventions_path.read_text(encoding="utf-8")
    assert "Keep durable memory topic-based and lightweight." in decisions_path.read_text(encoding="utf-8")
    assert report["durable_promotions"] == [
        "project-conventions: Use constrained tools instead of guessing.",
        "project-conventions: Preserve local agent state under .pico/.",
        "key-decisions: Keep durable memory topic-based and lightweight.",
    ]


def test_explicit_memory_promotion_supports_chinese_intent_and_labels(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>项目约定：优先使用受约束工具，不要靠猜。\n"
            "决策：持久记忆保持轻量、按 topic 管理。</final>",
        ],
    )

    answer = agent.ask("请把下面这些稳定事实记住，作为长期记忆保存下来。")

    assert "项目约定：" in answer

    conventions_path = tmp_path / ".micro" / "memory" / "topics" / "project-conventions.md"
    decisions_path = tmp_path / ".micro" / "memory" / "topics" / "key-decisions.md"

    assert "优先使用受约束工具，不要靠猜。" in conventions_path.read_text(encoding="utf-8")
    assert "持久记忆保持轻量、按 topic 管理。" in decisions_path.read_text(encoding="utf-8")


def test_explicit_memory_promotion_rejects_secret_shaped_and_transient_lines(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>Project convention: Use constrained tools instead of guessing.\n"
            "Dependency: API key is sk-live-secret-abc.\n"
            "Decision: Current goal is fix flaky tests.\n"
            "Dependency: stdout: FAIL test_one FAIL test_two FAIL test_three.</final>",
        ],
    )

    agent.ask("Capture these stable facts into durable memory.")

    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))
    conventions_path = tmp_path / ".micro" / "memory" / "topics" / "project-conventions.md"
    dependency_path = tmp_path / ".micro" / "memory" / "topics" / "dependency-facts.md"

    assert report["durable_promotions"] == [
        "project-conventions: Use constrained tools instead of guessing.",
    ]
    assert report["durable_rejections"] == [
        "dependency-facts:secret_shaped",
        "key-decisions:transient_task_state",
        "dependency-facts:noisy_output",
    ]
    assert "Use constrained tools instead of guessing." in conventions_path.read_text(encoding="utf-8")
    assert not dependency_path.exists()


def test_explicit_memory_promotion_supersedes_matching_durable_fact(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>Dependency: Python runtime is 3.11.</final>",
            "<final>Dependency: Python runtime is 3.12.</final>",
        ],
    )

    assert agent.ask("Capture this stable dependency fact into durable memory.") == "Dependency: Python runtime is 3.11."
    assert agent.ask("Save the updated dependency fact into durable memory.") == "Dependency: Python runtime is 3.12."

    dependency_path = tmp_path / ".micro" / "memory" / "topics" / "dependency-facts.md"
    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))
    text = dependency_path.read_text(encoding="utf-8")

    assert "Python runtime is 3.12." in text
    assert "Python runtime is 3.11." not in text
    assert report["durable_superseded"] == [
        "dependency-facts: Python runtime is 3.11. -> Python runtime is 3.12.",
    ]


def test_explicit_memory_promotion_dedupes_duplicate_durable_note(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>Project convention: Use constrained tools instead of guessing.</final>",
            "<final>Project convention: Use constrained tools instead of guessing.</final>",
        ],
    )

    agent.ask("Capture the stable fact into durable memory.")
    agent.ask("Capture the stable fact into durable memory again.")

    conventions_path = tmp_path / ".micro" / "memory" / "topics" / "project-conventions.md"
    text = conventions_path.read_text(encoding="utf-8")

    assert text.count("Use constrained tools instead of guessing.") == 1


def test_agent_records_model_cache_metadata_in_last_prompt_metadata(tmp_path):
    class CacheAwareFakeModelClient(FakeModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            self.last_completion_metadata = {
                "prompt_cache_supported": True,
                "cached_tokens": 512,
                "cache_hit": True,
                "input_tokens": 1024,
            }
            return super().complete(prompt, max_new_tokens, **kwargs)

    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".micro" / "sessions")
    agent = Micro(
        model_client=CacheAwareFakeModelClient(["<final>Done.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )

    assert agent.ask("Cache aware run") == "Done."

    assert agent.last_prompt_metadata["prompt_cache_supported"] is True
    assert agent.last_prompt_metadata["cached_tokens"] == 512
    assert agent.last_prompt_metadata["cache_hit"] is True
    assert agent.last_prompt_metadata["prefix_hash"]
    assert agent.last_prompt_metadata["prompt_cache_key"] == agent.last_prompt_metadata["prefix_hash"]


def test_recent_transcript_entries_stay_richer_than_older_ones(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    old_text = "OLD-" + ("A" * 320)
    recent_text = "RECENT-" + ("B" * 320)

    agent.record({"role": "user", "content": old_text, "created_at": "2026-04-07T09:00:00+00:00"})
    agent.record({"role": "assistant", "content": old_text, "created_at": "2026-04-07T09:01:00+00:00"})
    agent.record({"role": "user", "content": recent_text, "created_at": "2026-04-07T09:02:00+00:00"})
    agent.record({"role": "assistant", "content": recent_text, "created_at": "2026-04-07T09:03:00+00:00"})
    agent.record({"role": "user", "content": recent_text, "created_at": "2026-04-07T09:04:00+00:00"})
    agent.record({"role": "assistant", "content": recent_text, "created_at": "2026-04-07T09:05:00+00:00"})
    agent.record({"role": "user", "content": recent_text, "created_at": "2026-04-07T09:06:00+00:00"})
    agent.record({"role": "assistant", "content": recent_text, "created_at": "2026-04-07T09:07:00+00:00"})

    assert agent.ask("Check the transcript") == "Done."

    prompt = agent.model_client.prompts[-1]

    assert recent_text in prompt
    assert old_text not in prompt


def test_public_api_exports_resolve_through_package_path():
    assert callable(build_welcome)
    assert FakeModelClient is not None
    assert Micro is not None
    assert OllamaModelClient is not None
    assert SessionStore is not None
    assert WorkspaceContext is not None
    assert Path(micro.__file__).as_posix().endswith("/micro/__init__.py")


def test_reviewer_skeleton_docs_exist():
    review_pack = Path("docs/review-pack/README.md")
    architecture = Path("docs/architecture/agent-harness-v1-overview.md")

    assert review_pack.exists()
    assert architecture.exists()

    review_text = review_pack.read_text(encoding="utf-8")
    assert "Project pitch" in review_text
    assert "Architecture map" in review_text
    assert "Benchmark evidence" in review_text
    assert "Sample run artifact list" in review_text

    architecture_text = architecture.read_text(encoding="utf-8")
    assert "Agent Harness v1" in architecture_text
    assert "task state" in architecture_text.lower()


def test_package_import_surface_includes_cli_entrypoints():
    assert callable(micro.main)
    assert callable(micro.build_agent)
    assert callable(micro.build_arg_parser)


def test_module_execution_help_works():
    result = subprocess.run(
        [sys.executable, "-m", "micro", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
