"""Pico 性能评测 —— 通过 subprocess 调用 python -m pico"""
import json, os, subprocess, sys, time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent

def load_tasks():
    return json.loads((WORKSPACE / "benchmarks" / "performance_tasks.json").read_text(encoding="utf-8"))

def run_one(task, idx, total):
    print(f"\n{'='*60}\n[{idx}/{total}] {task['id']}\n{'='*60}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE)
    r = subprocess.run(
        [sys.executable, "-m", "pico", task["prompt"], "--max-steps", str(task["max_steps"]),
         "--approval", "auto", "--cwd", str(WORKSPACE)],
        cwd=WORKSPACE, capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0:
        print(f"  FAIL: {r.stderr[:200]}")
    elapsed = int((time.time() - t0) * 1000)

    runs = sorted((WORKSPACE / ".pico" / "runs").glob("*"), key=lambda p: p.stat().st_mtime)
    metrics = {"task_id": task["id"], "category": task["category"],
               "success": r.returncode == 0, "elapsed_ms": elapsed,
               "tool_count": 0, "thinking_triggered": False, "reminders": 0,
               "avg_prompt_chars": 0, "doom_blocks": 0}
    if runs:
        tp = runs[-1] / "trace.jsonl"
        if tp.exists():
            events = [json.loads(l) for l in tp.read_text(encoding="utf-8").splitlines()]
            metrics["tool_count"] = len([e for e in events if e["event"]=="tool_executed"])
            metrics["thinking_triggered"] = "thinking_completed" in [e["event"] for e in events]
            metrics["reminders"] = len([e for e in events if e["event"]=="reminder_injected"])
            metrics["doom_blocks"] = len([e for e in events if e["event"]=="tool_executed" and e.get("tool_error_code")=="doom_loop_blocked"])
            prompts = [e for e in events if e["event"]=="prompt_built"]
            if prompts: metrics["avg_prompt_chars"] = sum(p["prompt_metadata"]["prompt_chars"] for p in prompts)//len(prompts)
    print(f"  Tools:{metrics['tool_count']} Think:{metrics['thinking_triggered']} Remind:{metrics['reminders']} Doom:{metrics['doom_blocks']} Time:{elapsed}ms")
    return metrics

tasks = [t for t in load_tasks()["tasks"] if len(sys.argv)<=1 or sys.argv[1] in t["id"]]
all_metrics = [run_one(t, i+1, len(tasks)) for i, t in enumerate(tasks)]

tools = [m["tool_count"] for m in all_metrics]; chars = [m["avg_prompt_chars"] for m in all_metrics if m["avg_prompt_chars"]]
elapsed = [m["elapsed_ms"] for m in all_metrics]; think = sum(1 for m in all_metrics if m["thinking_triggered"])
remind = sum(m["reminders"] for m in all_metrics); doom = sum(m["doom_blocks"] for m in all_metrics)
report = {"total":len(all_metrics),"successful":sum(1 for m in all_metrics if m["success"]),
    "avg_tools":round(sum(tools)/len(tools),1) if tools else 0,
    "avg_prompt_chars":round(sum(chars)/len(chars)) if chars else 0,
    "avg_elapsed_ms":round(sum(elapsed)/len(elapsed)) if elapsed else 0,
    "thinking_rate":f"{think}/{len(all_metrics)}","total_reminders":remind,"total_doom_blocks":doom}
out = WORKSPACE/"artifacts"/"performance-benchmark.json"; out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
print(f"\n{'='*60}\nSuccess: {report['successful']}/{report['total']}\nAvg tools: {report['avg_tools']} | Prompt: {report['avg_prompt_chars']} | Time: {report['avg_elapsed_ms']}ms\nThinking: {report['thinking_rate']} | Reminders: {remind} | Doom: {doom}")
