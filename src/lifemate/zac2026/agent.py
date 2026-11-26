"""Run the LifeMate tool loop against one ToolEnv."""

import json
import re
import time
from types import SimpleNamespace

from .guard import ASK, Guard, UnverifiedReference
from .prompt import DEMO_HEADER, MAX_RESULT_CHARS, build_messages, build_tools, from_fn, to_fn


MAX_STEPS = 14
MAX_CALLS = 8


def _lenient_calls(text):
    """Parse a tool call returned as text by the local model."""
    out = []
    for i, blob in enumerate(re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text or "", re.S)):
        m = re.search(r'"name"\s*:\s*"([^"]+)"', blob)
        if not m:
            continue
        rest, args = blob[m.end():], "{}"
        j = rest.find("{")
        if j >= 0:
            depth = 0
            for k, ch in enumerate(rest[j:]):
                depth += (ch == "{") - (ch == "}")
                if depth == 0:
                    args = rest[j:j + k + 1]
                    break
        try:
            a = json.loads(args)
            a = a.get("arguments", a) if isinstance(a, dict) and set(a) <= {"arguments"} else a
            function = SimpleNamespace(name=m.group(1), arguments=json.dumps(a, ensure_ascii=False))
            out.append(SimpleNamespace(id=f"lenient{i}", function=function))
        except ValueError:
            continue
    return out


def _shrink_messages(messages, limit=600):
    """Drop few-shot examples and shorten tool messages after context overflow."""
    out = []
    for m in messages:
        if m["role"] == "system" and DEMO_HEADER in m["content"]:
            m = {**m, "content": m["content"].split(DEMO_HEADER)[0]}
        elif m["role"] == "tool" and len(m["content"]) > limit:
            m = {**m, "content": m["content"][:limit] + "…"}
        out.append(m)
    return out


def _chat(chat, messages, tools, deadline):
    try:
        return chat(messages, tools, deadline).choices[0].message, messages
    except Exception as exc:
        if "maximum context length" not in str(exc):
            raise
    for limit in (600, 200, 80):
        messages = _shrink_messages(messages, limit)
        try:
            return chat(messages, tools, deadline).choices[0].message, messages
        except Exception as exc:
            if "maximum context length" not in str(exc):
                raise
    return None, messages


def run_sample(row, env, specs, transcript, fewshot, whitelist, chat, fewshot_count, seconds):
    start = time.monotonic()
    diagnostics = {}
    decisions = []
    try:
        _tool_loop(row, env, specs, transcript, fewshot, whitelist, chat, fewshot_count, start + seconds, decisions)
    except TimeoutError:
        diagnostics["timeout"] = True
    except Exception as exc:
        diagnostics["error"] = repr(exc)[:300]
    return {
        "id": row["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
        "t_s": time.monotonic() - start, "guard_decisions": decisions, **diagnostics,
    }


def _tool_loop(row, env, specs, transcript, fewshot, whitelist, chat, fewshot_count, deadline, decisions=None):
    tools = build_tools(specs, row)
    query = transcript(row["audio"])
    user_turns = [transcript(h.get("audio")) for h in row["history"] if h["role"] == "user"]
    guard = Guard(specs, row, whitelist, " . ".join([*user_turns, query]), current_text=query)
    if decisions is not None:
        guard.decisions = decisions
    messages = build_messages(row, fewshot.topk(query, fewshot_count), transcript)
    validation_failures = 0
    for _ in range(MAX_STEPS):
        msg, messages = _chat(chat, messages, tools, deadline)
        if msg is None:
            return
        content = msg.content or ""
        calls = msg.tool_calls or _lenient_calls(content)
        if not calls:
            return
        messages.append({"role": "assistant", "content": content, "tool_calls": [
            {"id": call.id, "type": "function",
             "function": {"name": call.function.name, "arguments": call.function.arguments}}
            for call in calls
        ]})
        stop = False
        for call in calls:
            name = from_fn(call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
            except ValueError:
                args = None
            if stop:
                result = {"ok": False, "error": "Đã dừng, chưa thực thi."}
            elif not isinstance(args, dict):
                result = {"ok": False, "error": "arguments phải là JSON object; chưa thực thi"}
            else:
                if name == ASK and isinstance(args.get("tool_name"), str):
                    args["tool_name"] = from_fn(args["tool_name"])
                try:
                    args = guard.fix(name, args)
                except UnverifiedReference as exc:
                    result = {"ok": False, "error": str(exc)}
                    validation_failures += 1
                else:
                    result = guard.cached(name, args)
                    if result is None:
                        validation = env.validate(name, args)
                        if not validation["ok"]:
                            result = {"ok": False, "error": validation["error"], "note": "chưa thực thi; sửa tham số rồi gọi lại"}
                            validation_failures += 1
                        else:
                            if time.monotonic() >= deadline:
                                raise TimeoutError
                            validation_failures = 0
                            result = env.call(name, args)
                            guard.record_result(name, args, result)
                            stop = name == ASK or len(env.get_tool_calls()) >= MAX_CALLS
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)[:MAX_RESULT_CHARS]})
        if stop or validation_failures >= 4:
            return
