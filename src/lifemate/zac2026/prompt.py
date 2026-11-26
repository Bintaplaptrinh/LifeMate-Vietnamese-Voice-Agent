"""Build the LifeMate prompt, few-shot examples and OpenAI tool schemas."""

import copy
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .guard import ASK


MAX_RESULT_CHARS = 8000


def to_fn(name):
    return name.replace(".", "__", 1)


def from_fn(name):
    return name.replace("__", ".", 1)


def _clean_prop(p, required):
    p = copy.deepcopy(p)
    notes = []
    if p.pop("x-match", None) == "fuzzy":
        notes.append("văn bản tự do, viết ngắn gọn theo lời người dùng")
    ref = p.pop("x-ref", None)
    if ref:
        notes.append(f"id lấy từ {ref}, tra bằng tool đọc nếu chưa có trong lịch sử")
    if not required:
        notes.append("tuỳ chọn")
    for k in [k for k in p if k.startswith("x-")]:
        p.pop(k)
    if notes:
        p["description"] = "; ".join(filter(None, [p.get("description"), *notes]))
    return p


def to_openai_tool(spec):
    params = spec.get("parameters") or {}
    req = params.get("required", [])
    desc = spec.get("description", "")
    if params.get("x-one-of"):
        desc += " | Bắt buộc có một trong các nhóm: " + " / ".join(" + ".join(g) for g in params["x-one-of"])
    if spec.get("returns"):
        desc += f" | Trả về: {spec['returns']}"
    return {"type": "function", "function": {
        "name": to_fn(spec["name"]),
        "description": desc,
        "parameters": {"type": "object",
                       "properties": {k: _clean_prop(v, k in req) for k, v in (params.get("properties") or {}).items()},
                       "required": req}}}


def build_tools(specs, row):
    names = [t for a in row["context"]["installed_apps"].values() for t in a["tools"]] + [ASK]
    return [to_openai_tool(specs[n]) for n in names if n in specs]


class FewShotIndex:
    def __init__(self, rows):
        self.rows = rows
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.X = self.vec.fit_transform([r["audio_text"] for r in rows])

    def topk(self, text, k):
        if k <= 0:
            return []
        sims = linear_kernel(self.vec.transform([text]), self.X).ravel()
        return [self.rows[i] for i in sims.argsort()[::-1][:k]]


def render_demo(r):
    lines = [f"Người dùng: {r['audio_text']}"]
    if not r["label"]["tool_calls"]:
        lines.append("→ (không gọi tool nào)")
    for c, res in zip(r["label"]["tool_calls"], r["label"]["tool_results"]):
        lines.append(f"→ {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})")
        lines.append(f"   kết quả: {json.dumps(res, ensure_ascii=False)[:300]}")
    return "\n".join(lines)


SYSTEM_PROMPT = """Bạn là LifeMate, trợ lý ảo tiếng Việt điều khiển các app trên điện thoại của người dùng bằng cách gọi tool.

# Ngữ cảnh
- Thời gian hiện tại: {current_time} (múi giờ {timezone})
- Vị trí hiện tại: {location}
- Người dùng: {profile}

# Quy tắc (mọi tool call đều bị chấm điểm, call thừa hay sai đều mất điểm)
1. Chỉ xử lý YÊU CẦU CUỐI CÙNG. Các lượt trước đã thực hiện xong, không làm lại.
2. Tool đọc chỉ gọi khi thật sự cần: để lấy id (contact_id, device_id, bill_id...) chưa có trong lịch sử, để kiểm tra điều kiện, hoặc khi người dùng hỏi thông tin. KHÔNG đọc lại dữ liệu đã có trong lịch sử hội thoại. KHÔNG đọc "cho chắc".
3. Tham số tuỳ chọn: CHỈ điền khi người dùng nói rõ giá trị đó. Không tự thêm label, note, repeat, location, type, color, priority, payment...
4. Thiếu thông tin bắt buộc mà không suy ra được: làm hết phần làm được trước, rồi gọi sys__ask_missing_info(tool_name="app.tool", params=[ĐÚNG MỘT tên tham số thiếu]) làm call CUỐI CÙNG, chỉ một lần. Thiếu id (vd. tên liên hệ) → tra trước (contacts.search...); không tìm thấy hoặc có nhiều kết quả trùng mà không phân biệt được → ask (nhóm x-one-of thì dùng tham số đầu, vd. contact_id). Thiếu giá trị (số tiền, giờ, nội dung) → ask luôn, không cần tra.
5. Yêu cầu có điều kiện ("nếu mai mưa thì...") → kiểm tra điều kiện (nếu lịch sử chưa cho biết) rồi chỉ thực hiện ĐÚNG MỘT nhánh phù hợp với kết quả.
6. Không gọi tool nào khi: yêu cầu không làm được bằng các app đã cài, là chuyện phiếm/chào/cảm ơn, hoặc người dùng tự huỷ ("à thôi khỏi").
7. Định dạng: thời điểm "YYYY-MM-DDTHH:MM:SS+07:00" (đủ giây và múi giờ), ngày "YYYY-MM-DD", số là số nguyên không dấu chấm ("hai triệu tám" = 2800000). Từ khoá tìm kiếm giữ nguyên tên người dùng nói, viết hoa như tên riêng.
8. Gọi từng tool, đợi kết quả rồi mới quyết định bước tiếp. Xong việc thì trả lời ngắn gọn bằng tiếng Việt và không gọi thêm tool.{demo_block}"""

DEMO_HEADER = "\n\n# Ví dụ tương tự (từ các mẫu khác, id và dữ liệu khác với mẫu hiện tại, chỉ để tham khảo cách gọi)\n"


def build_messages(row, demos, transcript, *, system_prompt=SYSTEM_PROMPT, **context):
    ctx = row["context"]
    sys_msg = system_prompt.format(
        current_time=ctx["current_time"], timezone=ctx.get("timezone", ""),
        location=json.dumps(ctx.get("location"), ensure_ascii=False),
        profile=json.dumps(ctx.get("user_profile"), ensure_ascii=False),
        demo_block=DEMO_HEADER + "\n\n".join(render_demo(d) for d in demos) if demos else "",
        **context)
    msgs = [{"role": "system", "content": sys_msg}]
    for i, h in enumerate(row["history"]):
        if h["role"] == "user":
            msgs.append({"role": "user", "content": transcript(h.get("audio"))})
            continue
        calls = h.get("tool_calls") or []
        if calls:
            ids = [f"h{i}_{j}" for j in range(len(calls))]
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": to_fn(c["name"]), "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                for cid, c in zip(ids, calls)]})
            results = h.get("tool_results") or [{}] * len(calls)
            for cid, res in zip(ids, results):
                msgs.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(res, ensure_ascii=False)[:MAX_RESULT_CHARS]})
        msgs.append({"role": "assistant", "content": h.get("text", "")})
    msgs.append({"role": "user", "content": transcript(row["audio"])})
    return msgs
