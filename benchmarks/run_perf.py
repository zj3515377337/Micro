import json, sys, time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent

def load_tasks():
    return json.loads((WORKSPACE / "benchmarks" / "performance_tasks.json").read_text(encoding="utf-8"))

def run_one(task, idx, total):
    from pico.cli import build_agent, build_arg_parser
    print(f"\n{'='*60}\n[{idx}/{total}] {task['id']}\n{'='*60}")
    parser = build_arg_parser()
    args = parser.parse_args([task["prompt"], "--max-steps", str(task["max_steps"]), "--approval", "auto", "--cwd", str(WORKSPACE)])
    agent = build_agent(args)
    t0 = time.time()
    try:
        result = agent.ask(task["prompt"])
        elapsed = int((time.time() - t0) * 1000)
    except Exception as e:
        result = f"ERROR: {e}"
        elapsed = int((time.time() - t0) * 1000)

    metrics = {"task_id": task["id"], "category": task["category"],
               "result": result[:200], "elapsed_ms": elapsed,
               "success": not result.startswith("ERROR"),
               "tool_count": 0, "thinking_triggered": False, "reminders": 0,
               "avg_prompt_chars": 0, "doom_blocks": 0}

    runs = sorted((WORKSPACE / ".pico" / "runs").glob("*"), key=lambda p: p.stat().st_mtime)
    if runs:
        trace_path = runs[-1] / "trace.jsonl"
        if trace_path.exists():
            events = [json.loads(l) for l in trace_path.read_text().splitlines()]
            metrics["tool_count"] = len([e for e in events if e["event"] == "tool_executed"])
            metrics["thinking_triggered"] = "thinking_completed" in [e["event"] for e in events]
            metrics["reminders"] = len([e for e in events if e["event"] == "reminder_injected"])
            prompts = [e for e in events if e["event"] == "prompt_built"]
            if prompts:
                metrics["avg_prompt_chars"] = sum(p["prompt_metadata"]["prompt_chars"] for p in prompts) // len(prompts)
            for e in events:
                if e["event"] == "tool_executed" and e.get("tool_error_code") == "doom_loop_blocked":
                    metrics["doom_blocks"] += 1

    print(f"  Tools:{metrics['tool_count']} Think:{metrics['thinking_triggered']} Remind:{metrics['reminders']} Doom:{metrics['doom_blocks']} Time:{elapsed}ms")
    return metrics

def main():
    tasks = load_tasks()["tasks"]
    if len(sys.argv) > 1:
        tasks = [t for t in tasks if sys.argv[1] in t["id"]]
    all_metrics = [run_one(t, i+1, len(tasks)) for i, t in enumerate(tasks)]
    
    tools = [m["tool_count"] for m in all_metrics]
    prompts = [m["avg_prompt_chars"] for m in all_metrics if m["avg_prompt_chars"] > 0]
    elapsed = [m["elapsed_ms"] for m in all_metrics]
    thinking = sum(1 for m in all_metrics if m["thinking_triggered"])
    reminders = sum(m["reminders"] for m in all_metrics)
    doom = sum(m["doom_blocks"] for m in all_metrics)
    success = sum(1 for m in all_metrics if m["success"])

    report = {
        "total": len(all_metrics), "successful": success,
        "avg_tools": round(sum(tools)/len(tools), 1),
        "avg_prompt_chars": round(sum(prompts)/len(prompts)) if prompts else 0,
        "avg_elapsed_ms": round(sum(elapsed)/len(elapsed)),
        "thinking_rate": f"{thinking}/{len(all_metrics)}",
        "total_reminders": reminders,
        "total_doom_blocks": doom,
        "tasks": all_metrics,
    }
    
    out = WORKSPACE / "artifacts" / "performance-benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}\nPERFORMANCE BENCHMARK RESULTS\n{'='*60}")
    print(f"  Success: {success}/{len(all_metrics)}")
    print(f"  Avg tools: {report['avg_tools']} | Avg prompt: {report['avg_prompt_chars']} chars | Avg time: {report['avg_elapsed_ms']}ms")
    print(f"  Thinking: {report['thinking_rate']} | Reminders: {reminders} | Doom blocks: {doom}")
    print(f"\n  Artifact: {out}")

if __name__ == "__main__":
    main()
