"""Domain abstention, lookup recovery, clarification and execution."""

import json
import math
import re
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ..zac2026.guard import ASK, _spoken
from ..zac2026.prompt import from_fn


_CHITCHAT = re.compile(
    r"^(?:(?:da|a|uhm|vang|u|o) )?(?:cam on(?: ban| em| anh| chi)?(?: nhieu)?|"
    r"xin chao|chao ban|tam biet|khong co gi|(?:a )?thoi(?: khoi| bo qua| khong can(?: nua)?)?|"
    r"khong can nua|huy bo|dung lam gi(?: ca)?|bo qua di)(?: nhe| nha| a| di)*$"
)
_CANCEL_TAIL = re.compile(r"[,;.!?]\s*(?:a thoi|ma thoi|thoi khoi|thoi bo qua|thoi dung)"
                          r"(?:,?\s+(?:khoi|khong can|dung|bo qua|huy|lam|nua|nhe|nha|di|a))*$")
_NEGATIVE_REQUEST = re.compile(
    r"^(?:dung|khong can|khoi) (?:goi|nhan tin|bat|tat|thanh toan|mua|dat|chuyen|"
    r"gui|xoa|huy|nap|tao|doi|mo|dong|luu|nhac)\b"
)
_FOLLOW_UP = re.compile(r"\b(?:cai do|mon do|gio do|nguoi do|nhu cu|nhu truoc|"
                        r"cai day|hom day|doi sang|doi thanh|thay bang|chuyen sang|"
                        r"them nua|van vay|lui\b.*\bsang)\b")
_STOP_WORDS = frozenset(
    "cho toi minh em anh chi co voi giup nhe nha mot cai nay do lay gi la nao bao hay xem hoi "
    "can muon nho thu de o ve luon roi ra hoi gi khong duoc may giup du di con nay so".split()
)
_APP_TERMS = {
    "alarm": "bao thuc hen gio danh thuc", "reminders": "nhac nho ghi nho",
    "calendar": "lich hen lich hop su kien cuoc hen", "smart_home": "den quat dieu hoa may lanh rem o cam",
    "phone": "goi dien thoai cuoc goi so dien thoai", "sms": "tin nhan nhan tin sms",
    "chat_ott": "chat nhan tin zalo cuoc goi", "contacts": "danh ba lien he",
    "banking": "ngan hang so du tai khoan chuyen tien gui tiet kiem",
    "ewallet": "vi dien tu chuyen tien momo nap tien", "food_delivery": "dat mon do an nha hang quan an",
    "grocery": "di cho sieu thi thuc pham gio hang", "ecommerce": "mua hang thuong mai dien tu gio hang",
    "ride_hailing": "goi xe dat xe grab xe om", "weather": "thoi tiet mua nang nhiet do du bao",
    "notes": "ghi chu ghi lai luu lai", "tasks": "viec can lam cong viec to do",
    "translator": "dich tieng anh tieng viet ngon ngu phien dich",
    "parking": "gui xe bai do xe cho gui xe", "stocks": "co phieu chung khoan gia co phieu",
}


def _terms(text):
    return {w for w in re.findall(r"[a-z]{2,}", text) if w not in _STOP_WORDS}


class AbstentionSkill:
    """Reject clear cancellations and expose weak catalog overlap for diagnostics."""

    def __init__(self, specs):
        self.names = [name for name in specs if name != ASK]
        self.columns = {name: i for i, name in enumerate(self.names)}
        if not self.names:
            return
        docs = []
        for name in self.names:
            spec = specs[name]
            app = name.partition(".")[0]
            docs.append(_spoken(" ".join([
                name.replace(".", " ").replace("_", " "), spec.get("description") or "",
                _APP_TERMS.get(app, ""),
            ])))
        self.terms = [_terms(doc) for doc in docs]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(docs)
        self.analyze = self.vectorizer.build_analyzer()

    def check(self, text, installed, history):
        query = re.sub(r"[,;.!?]+$", "", text.strip()).strip()
        columns = [self.columns[name] for name in installed if name in self.columns]
        if not columns or not query:
            return "no_domain", None
        if (_CHITCHAT.fullmatch(query) or _CANCEL_TAIL.search(query)
                or (_NEGATIVE_REQUEST.search(query)
                    and not re.search(r"[,;]|\b(?:ma|nhung|thay vao do|roi|sau do|con|chi can)\b", query))):
            return "cancel_or_chitchat", None
        terms = _terms(query)
        if len(terms) < 4 or (history and (len(query.split()) <= 12 or _FOLLOW_UP.search(query))):
            return None, None
        sims = linear_kernel(self.vectorizer.transform([query]), self.matrix[columns]).ravel()
        # Unseen n-grams vanish from the TF-IDF vector. Discount cosine similarity
        # when most of the original utterance was discarded.
        grams = set(self.analyze(query))
        coverage = sum(g in self.vectorizer.vocabulary_ for g in grams) / max(1, len(grams))
        score = float(sims.max()) * math.sqrt(coverage)
        matches = [terms & self.terms[i] for i in columns]
        keyword = any(len(m) >= 2 or any(len(w) >= 4 for w in m) for m in matches)
        if score < 0.12 and not keyword:
            return "low_domain_match", score
        return None, score


class DiscoverySkill:
    @staticmethod
    def read_tools(name, error, specs, installed):
        app, _, collection = error.reference.partition(".")
        reads = sorted(tool for tool in installed if tool != name
                       and tool.startswith(app + ".") and specs[tool].get("kind") == "read")
        # ToolEnv exposes the expected collection in the read tool's returns metadata.
        matching = [name for name in reads if re.search(
            rf"\b{re.escape(collection)}\b", specs[name].get("returns") or "")]
        return matching + [name for name in reads if name not in matching]

    @classmethod
    def feedback(cls, name, error, specs, installed):
        return {
            "ok": False, "error": f"Tham số {error.key} chưa có bằng chứng từ {error.reference}.",
            "skill": "discovery",
            "read_tools": cls.read_tools(name, error, specs, installed),
            "note": "Chưa thực thi. Có thể tra cứu hoặc dùng cách khác có đủ dữ liệu. "
                    "Hoàn thành phần độc lập trước, rồi hỏi nếu vẫn chưa xác định được.",
        }


class ClarificationSkill:
    FIELD_LABELS = {
        "time": "giờ", "start": "thời gian bắt đầu", "end": "thời gian kết thúc", "date": "ngày",
        "content": "nội dung", "body": "nội dung", "title": "tiêu đề", "subject": "chủ đề",
        "amount": "số tiền", "price": "giá", "quantity": "số lượng", "passengers": "số hành khách",
        "seats": "số ghế", "phone": "số điện thoại", "phone_number": "số điện thoại",
        "contact_id": "người cần liên hệ", "to": "người nhận", "chat_id": "cuộc trò chuyện",
        "device_id": "thiết bị cần dùng", "event_id": "sự kiện", "alarm_id": "báo thức",
        "items": "món và số lượng", "restaurant_id": "tên quán", "address": "địa chỉ",
        "pickup": "điểm đón", "dropoff": "điểm đến", "vehicle": "loại xe",
    }

    @staticmethod
    def arguments(name, fields, specs, installed):
        name = from_fn(name) if isinstance(name, str) else None
        if ASK not in installed or name not in installed or name == ASK or not isinstance(fields, (list, tuple)):
            return None
        schema = specs[name].get("parameters") or {}
        keys = set(schema.get("required") or ())
        keys.update(key for group in schema.get("x-one-of") or () for key in group)
        keys.update(key for key, prop in (schema.get("properties") or {}).items() if prop.get("x-ref"))
        if not fields or not all(isinstance(k, str) and k in keys for k in fields):
            return None
        return {"tool_name": name, "params": list(dict.fromkeys(fields))}

    @classmethod
    def question(cls, fields):
        labels = [cls.FIELD_LABELS.get(key, key.replace("_", " ")) for key in fields]
        return "Bạn cho mình biết " + ", ".join(labels) + " được không?"


class ExecutionSkill:
    @staticmethod
    def execute(env, guard, name, args, deadline):
        cached = guard.cached(name, args)
        if cached is not None:
            return cached, False
        validation = env.validate(name, args)
        if not validation["ok"]:
            return {"ok": False, "error": validation["error"], "note": "Chưa thực thi; sửa tham số rồi gọi lại."}, False
        if time.monotonic() >= deadline:
            raise TimeoutError
        result = env.call(name, args)
        guard.record_result(name, args, result)
        return result, True

    @staticmethod
    def encode(result, limit):
        return json.dumps(result, ensure_ascii=False)[:limit]
