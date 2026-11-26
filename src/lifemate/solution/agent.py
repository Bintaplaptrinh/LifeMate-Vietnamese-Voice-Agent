"""Run one v4 sample without letting rejected or repeated calls change ToolEnv."""

import json
import time

from ..zac2026.agent import MAX_CALLS, MAX_STEPS, _chat, _lenient_calls
from ..zac2026.guard import ASK, UnverifiedReference
from ..zac2026.prompt import MAX_RESULT_CHARS, build_tools, from_fn
from .prompt import build_messages
from .skills import ClarificationSkill, DiscoverySkill, ExecutionSkill


def run_sample(row, env, *, specs, transcript, fewshot, guard_factory, abstention, chat, fewshot_count, seconds):
    start = time.monotonic()
    trace = {"guard_decisions": [], "skills": []}
    try:
        _tool_loop(row, env, specs, transcript, fewshot, guard_factory, abstention, chat, fewshot_count,
                   start + seconds, trace)
    except TimeoutError:
        trace["timeout"] = True
    except Exception as exc:
        trace["error"] = repr(exc)[:300]
    return {
        "id": row["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
        "t_s": time.monotonic() - start, **trace,
    }


def _tool_loop(row, env, specs, transcript, fewshot, guard_factory, abstention, chat, fewshot_count, deadline, trace):
    tools = build_tools(specs, row)
    installed = {from_fn(tool["function"]["name"]) for tool in tools}
    query = transcript(row["audio"])
    user_turns = [transcript(turn["audio"]) for turn in row["history"]
                  if turn["role"] == "user" and turn.get("audio")]
    guard = guard_factory(specs, row, text=" . ".join([*user_turns, query]), current_text=query)
    guard.decisions = trace["guard_decisions"]
    reason, score = abstention.check(guard.current_text, installed, row["history"])
    trace["domain_similarity"] = round(score, 4) if score is not None else None
    if reason == "low_domain_match":
        trace["domain_assessment"] = reason

    def abstain():
        trace["skills"].append({"skill": "abstention", "reason": reason})
        if reason == "cancel_or_chitchat":
            trace["response"] = "Được, bạn cứ nói nếu cần mình giúp thêm."
        else:
            trace["response"] = "Mình chưa thể làm việc này với các ứng dụng hiện có."

    def clarify(name, fields):
        trace["skills"].append({"skill": "clarification", "tool": name, "fields": list(fields)})
        trace["response"] = ClarificationSkill.question(fields)
        args = ClarificationSkill.arguments(name, fields, specs, installed)
        if args is not None and len(env.get_tool_calls()) < MAX_CALLS:
            result, _ = ExecutionSkill.execute(env, guard, ASK, args, deadline)
            return result
        return {"ok": False, "note": trace["response"]}

    if reason in {"no_domain", "cancel_or_chitchat"}:
        abstain()
        return

    messages = build_messages(row, fewshot.topk(query, fewshot_count), transcript)
    pending = {}
    stalled = 0
    for _ in range(MAX_STEPS):
        msg, messages = _chat(chat, messages, tools, deadline)
        if msg is None:
            break
        content = msg.content or ""
        calls = msg.tool_calls or _lenient_calls(content)
        if not calls:
            if content:
                trace["response"] = content
            break

        messages.append({"role": "assistant", "content": content, "tool_calls": [
            {"id": call.id, "type": "function",
             "function": {"name": call.function.name, "arguments": call.function.arguments}}
            for call in calls
        ]})
        progress = False
        blocked = False
        requested_question = None
        for call in calls:
            name = from_fn(call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
            except ValueError:
                args = None
            if blocked:
                result = {"ok": False, "note": "Chưa thực thi vì call trước chưa hoàn tất. "
                          "Nếu call này độc lập, có thể gửi lại ở lượt kế tiếp."}
            elif name not in installed:
                result = {"ok": False, "error": "Công cụ không có trong danh sách đã cài; chưa thực thi."}
            elif not isinstance(args, dict):
                result = {"ok": False, "error": "arguments phải là JSON object; chưa thực thi."}
            elif name == ASK:
                args = ClarificationSkill.arguments(args.get("tool_name"), args.get("params"), specs, installed)
                if args is None:
                    result = {"ok": False, "error": "Dùng tool_name và params gồm tên trường thiếu trong schema."}
                else:
                    requested_question = args
                    result = {"ok": True, "note": "Câu hỏi sẽ được gửi sau các thao tác độc lập trong lượt này."}
            else:
                try:
                    args = guard.fix(name, args)
                except UnverifiedReference as exc:
                    result = DiscoverySkill.feedback(name, exc, specs, installed)
                    trace["skills"].append({"skill": "discovery", "tool": name, "field": exc.key,
                                            "read_tools": result["read_tools"]})
                    pending[name] = [exc.key]
                except ValueError:
                    result = {"ok": False, "error": "Giá trị không đúng định dạng; chưa thực thi."}
                else:
                    result, executed = ExecutionSkill.execute(env, guard, name, args, deadline)
                    if executed:
                        progress = True
                        trace["skills"].append({"skill": "execution", "tool": name})
                    elif "note" in result and result.get("ok") is True:
                        trace["skills"].append({"skill": "execution", "tool": name, "cached": True})
                    if result.get("ok") is True:
                        pending.pop(name, None)
                        if requested_question and requested_question["tool_name"] == name:
                            requested_question = None
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": ExecutionSkill.encode(result, MAX_RESULT_CHARS)})
            blocked = blocked or result.get("ok") is False or result.get("done") is False
            if len(env.get_tool_calls()) >= MAX_CALLS:
                return
        if requested_question:
            clarify(requested_question["tool_name"], requested_question["params"])
            return
        stalled = 0 if progress else stalled + 1
        if stalled >= 3:
            break
    if pending:
        name, fields = next(iter(pending.items()))
        clarify(name, fields)
