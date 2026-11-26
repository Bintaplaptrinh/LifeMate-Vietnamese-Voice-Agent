"""Normalize arguments and deduplicate calls in one sample."""

import copy
import json
import re
from collections import Counter
from datetime import datetime, timedelta

from .vnnum import fold, normalize_spoken


_DT = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$")
_EMPTY = (None, "", [], {})
_CLOCK = re.compile(r"(?<![\w:])(\d{1,2})(?::(\d{2})(?::\d{2})?|\s+gio(?:\s+(\d{1,2})\s+phut)?)\b")
ASK = "sys.ask_missing_info"
GUARD_VERSION = "Guard V3"


def canon_value(v, p, tz):
    t = p.get("type")
    if isinstance(v, str):
        s = v.strip()
        m = _DT.match(s)
        if m and p.get("format") != "date":
            d, hh, mm, ss, off = m.groups()
            dt = datetime.fromisoformat(f"{d}T{int(hh):02d}:{mm}:{ss or '00'}")
            if off:
                off = "+00:00" if off == "Z" else (off if ":" in off else off[:3] + ":" + off[3:])
                dt = datetime.fromisoformat(dt.isoformat() + off).astimezone(tz)
            else:
                dt = dt.replace(tzinfo=tz)
            return dt.isoformat()
        if t == "integer" and re.fullmatch(r"-?\d[\d.,]*", s) and re.fullmatch(r"-?\d+", s.replace(".", "").replace(",", "")):
            return int(s.replace(".", "").replace(",", ""))
        for e in p.get("enum") or []:
            if isinstance(e, str) and e.lower() == s.lower():
                return e
    if t == "integer" and isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def opt_whitelist(rows, specs, min_rate=0.10):
    """{tool: {optional được giữ}} cho tool có trong label của rows (chỉ dùng rows không thuộc dev khi eval)."""
    n, f = Counter(), Counter()
    for r in rows:
        for c in r["label"]["tool_calls"]:
            n[c["name"]] += 1
            f.update((c["name"], k) for k in c["arguments"])
    return {name: {k for k in ((specs[name].get("parameters") or {}).get("properties") or {}) if f[(name, k)] / n[name] >= min_rate}
            for name in n if name in specs}


def _spoken(text):
    return fold(normalize_spoken(text))


def _said(v, text, *, normalized=False):
    if isinstance(v, list):
        return bool(v) and all(_said(item, text, normalized=normalized) for item in v)
    if not isinstance(v, (str, int, float)) or isinstance(v, bool):
        return False
    value = _spoken(str(v))
    text = text if normalized else _spoken(text)
    if not value:
        return False
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        value = value.rstrip("0").rstrip(".") if "." in value else value
        return re.search(r"(?<![\w+:-])(?<!\d\.)" + re.escape(value)
                         + r"(?:\.0+)?(?!\w|[.:-]\d)", text) is not None
    return re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text) is not None


_DEVICE_TYPES = {
    "light": ("đèn",), "fan": ("quạt",), "climate": ("điều hòa", "điều hoà", "máy lạnh"),
    "curtain": ("rèm",), "plug": ("ổ cắm",), "tv": ("tivi", "ti vi", "tv"),
}

_POWER = {"on": r"(?:bat|mo)", "off": r"tat"}
_SWITCH_PARAMS = {
    "power": _POWER,
    "enabled": {True: _POWER["on"], False: _POWER["off"]},
    "video": {True: r"(?:video|co hinh)", False: r"(?:thoai|khong hinh)"},
    "missed_only": {True: r"(?:goi nho|bo lo)"},
    "unread_only": {True: r"chua doc"},
}
_RELATIVE_DAY = {
    "ngay kia": 2, "ngay mai": 1, "mai": 1,
    "hom nay": 0, "sang nay": 0, "trua nay": 0, "chieu nay": 0, "toi nay": 0,
    "hom qua": -1,
}
_DAY_CUE = re.compile(
    r"\b(?:ngay kia|ngay mai|mai|hom nay|sang nay|trua nay|chieu nay|toi nay|hom qua"
    r"|ngay mot|(?:sang|trua|chieu|toi) mot)\b"
)
_DATE_CONTEXT = re.compile(
    r"\b(?:(?:tuan|thang) (?:nay|sau|toi|truoc)|thu (?:hai|ba|tu|nam|sau|bay|[2-7])"
    r"|chu nhat|(?:hom|ngay|mung|thang) \d+|hom do|ngay do|hom truoc|hom sau)\b"
)
_TIME_SHIFT = re.compile(
    r"\b(?:(?:som|muon|truoc|sau|them|lui)\s+(?:hon\s+|them\s+)?(?:\d+|nua)\s+(?:tieng|phut|gio)"
    r"|(?:dung|y) gio (?:do|ay)|(?:gio|luc) (?:hen|cu|moi))\b"
)
_NEGATED = re.compile(r"\b(?:dung|khong|khoi)(?: can| phai)?\s*$")

# Some tools return a singular record or the generic App.create/patch "item".
_ITEM_COLLECTIONS = {
    "alarm.create": "alarms", "alarm.update": "alarms",
    "reminders.create": "reminders", "reminders.complete": "reminders",
    "calendar.create_event": "events", "calendar.update_event": "events",
    "notes.create": "notes", "notes.append": "notes",
    "tasks.add": "tasks", "tasks.complete": "tasks",
    "contacts.update": "contacts",
    "smart_home.light_set": "devices", "smart_home.fan_set": "devices",
    "smart_home.climate_set": "devices", "smart_home.switch_set": "devices",
    "ride_hailing.book": "rides", "ride_hailing.cancel": "rides",
    "shopping_list.add": "items", "shopping_list.check": "items",
    "pantry.consume": "items",
    "medication.add": "schedules", "video_meeting.create": "meetings",
}
_SINGULAR_FIELDS = {"emails": "email", "recipes": "recipe", "rides": "ride",
                    "sessions": "session", "bookings": "booking", "orders": "order", "rules": "rule"}
_CONTACT_FIELDS = {
    "phone.recent_calls": "calls", "phone.call": "call",
    "sms.read_recent": "messages", "sms.send": "message",
}


class UnverifiedReference(ValueError):
    def __init__(self, key, value, reference):
        self.key, self.reference = key, reference
        app = reference.split(".", 1)[0]
        super().__init__(f"Tham số {key}='{value}' chưa được xác thực. Hãy gọi tool tìm kiếm/danh sách "
                         f"của ứng dụng {app} trước để lấy ID chính xác.")


def _reference_records(tool, result, collection):
    for field in (collection, _SINGULAR_FIELDS.get(collection),
                  "item" if _ITEM_COLLECTIONS.get(tool) == collection else None):
        value = result.get(field)
        if isinstance(value, list):
            yield from value
        elif isinstance(value, dict):
            yield value


def _reference_ids(tool, result, reference):
    app, _, collection = reference.partition(".")
    if tool.split(".", 1)[0] == app:
        for record in _reference_records(tool, result, collection):
            if isinstance(record, dict):
                yield record.get("id")
    if reference == "contacts.contacts" and tool in _CONTACT_FIELDS:
        # Call/SMS records carry the contact foreign key as a dedicated field.
        records = result.get(_CONTACT_FIELDS[tool])
        for record in records if isinstance(records, list) else [records]:
            if isinstance(record, dict):
                yield record.get("contact_id")
                if isinstance(record.get("contact"), dict):
                    yield record["contact"].get("id")


def _field_values(value, field):
    """Collect only the named field, never arbitrary strings from a tool response."""
    if isinstance(value, dict):
        if field in value:
            yield value[field]
        for child in value.values():
            yield from _field_values(child, field)
    elif isinstance(value, list):
        for child in value:
            yield from _field_values(child, field)


class Guard:
    """Trạng thái guard của một mẫu: múi giờ, lời người dùng (để xét optional), các read trong history."""

    def __init__(self, specs, row, whitelist, text, current_text=None):
        self.specs, self.whitelist = specs, whitelist
        self.text = _spoken(text)
        current = text if current_text is None else current_text
        self.current_text = _spoken(current)
        self.current_times = {(int(h), int(mm or minutes or 0))
                              for h, mm, minutes in _CLOCK.findall(self.current_text)}
        self.profile = row["context"].get("user_profile") or {}
        self.evidence = []
        self.decisions = []
        self.now = datetime.fromisoformat(row["context"]["current_time"])
        self.tz = self.now.tzinfo
        self.seen = {}                                          # key -> kết quả của call đã thực thi
        self.hist = {}                                          # key -> kết quả của read trong history
        for h in row["history"]:
            for c, res in zip(h.get("tool_calls") or [], h.get("tool_results") or []):
                self.observe(c["name"], res, "history_tool_result")
                if self.kind(c["name"]) == "read":
                    # Retain the original read scope. Filtering historical
                    # arguments could turn a narrow read into a broad cache hit.
                    props = (specs.get(c["name"], {}).get("parameters") or {}).get("properties", {})
                    normalized = {k: canon_value(v, props.get(k, {}), self.tz) for k, v in c["arguments"].items()}
                    self.hist[self.key(c["name"], normalized)] = res
                else:
                    self.invalidate(c["name"])

    def observe(self, name, result, source="tool_result"):
        if isinstance(result, dict) and result.get("ok") is True and result.get("done") is not False:
            self.evidence.append((name, copy.deepcopy(result), source))

    def source(self, name, key, value, prop):
        if value in _EMPTY:
            return None
        if _said(value, self.current_text, normalized=True):
            return "current_user"
        if name.startswith("smart_home.") and key == "type":
            aliases = _DEVICE_TYPES.get(value, ()) if isinstance(value, str) else ()
            if any(_said(alias, self.current_text, normalized=True) for alias in aliases):
                return "current_user:device_type"
        if _said(value, self.text, normalized=True):
            return "user_history"
        reference = prop.get("x-ref")
        if reference:
            for tool, result, origin in reversed(self.evidence):
                if any(candidate == value for candidate in _reference_ids(tool, result, reference)):
                    return origin + ":" + tool + ":" + reference.partition(".")[2]
            return None
        if key in ("address", "pickup", "location"):
            fields = (key, "home_address", "work_address")
            if any(value == candidate for field in fields for candidate in _field_values(self.profile, field)):
                return "user_profile"
        return None

    def inference(self, name, key, value, args):
        if key not in self.whitelist.get(name, ()):
            return None
        if isinstance(value, (str, bool)):
            cue = _SWITCH_PARAMS.get(key, {}).get(value)
            if cue and any(not _NEGATED.search(self.current_text[:m.start()])
                           for m in re.finditer(rf"\b(?:{cue})\b", self.current_text)):
                return "whitelist:action"
        if not isinstance(value, str):
            return None
        stamp = _DT.fullmatch(value)
        if not (stamp or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)):
            return None
        date = value[:10]
        days = {_RELATIVE_DAY.get(m[0], 2) for m in _DAY_CUE.finditer(self.current_text)}
        date_context = _DATE_CONTEXT.search(self.current_text)
        time_shift = _TIME_SHIFT.search(self.current_text)
        if not date_context and not (time_shift and not days):
            valid_dates = {(self.now.date() + timedelta(days=d)).isoformat() for d in days or {0, 1}}
            if date not in valid_dates:
                return None
        if not stamp:
            return "whitelist:relative_time" if days or date_context else None
        clock = (int(stamp[2]), int(stamp[3]))
        if clock in self.current_times or time_shift:
            return "whitelist:relative_time"
        if key == "end" and name == "calendar.create_event":
            start = args.get("start")
            if isinstance(start, str) and _DT.fullmatch(start):
                if ((int(start[11:13]), int(start[14:16])) in self.current_times
                        and datetime.fromisoformat(start) + timedelta(hours=1) == datetime.fromisoformat(value)):
                    return "whitelist:default_duration"
        return None

    def kind(self, name):
        return (self.specs.get(name) or {}).get("kind")

    @staticmethod
    def key(name, args):
        return json.dumps([name, args], sort_keys=True, ensure_ascii=False)

    def fix(self, name, args):
        schema = (self.specs.get(name) or {}).get("parameters") or {}
        props, req = schema.get("properties") or {}, set(schema.get("required") or [])
        args = {k: v for k, v in args.items() if v not in _EMPTY or k in req or props.get(k, {}).get("x-ref")}
        args = {k: canon_value(v, props.get(k, {}), self.tz) for k, v in args.items()}
        if name in self.specs:
            kept = {}
            groups = schema.get("x-one-of") or []
            alternative = next((set(group) for group in groups if all(args.get(k) not in _EMPTY for k in group)), set())
            for key, value in args.items():
                prop = props.get(key, {})
                source = self.source(name, key, value, prop)
                reference = prop.get("x-ref")
                blocked = bool(reference) and source is None
                necessary = key in req or key in alternative
                inference = None if source or blocked or necessary else self.inference(name, key, value, args)
                keep = not blocked and bool(source or necessary or inference)
                self.decisions.append({"tool": name, "parameter": key, "value": copy.deepcopy(value),
                                       "source": source or inference or ("unverified_reference" if blocked else "model_inference"),
                                       "kept": keep})
                if blocked:
                    raise UnverifiedReference(key, value, reference)
                if keep:
                    kept[key] = value
            args = kept
        return args

    def invalidate(self, name):
        app = name.split(".", 1)[0]
        for cache in (self.seen, self.hist):
            for k in list(cache):
                if json.loads(k)[0].split(".", 1)[0] == app:
                    del cache[k]

    def record_result(self, name, args, result):
        if self.kind(name) == "read":
            self.seen[self.key(name, args)] = result
        else:
            self.invalidate(name)
        self.observe(name, result)

    def cached(self, name, args):
        """Kết quả có sẵn (không gọi env) nếu call trùng call đã gọi hoặc read trùng history; None nếu phải gọi."""
        k = self.key(name, args)
        if k in self.seen:
            return {**self.seen[k], "note": "call này đã gọi rồi, đây là kết quả cũ; không gọi lại"}
        if self.kind(name) == "read" and k in self.hist:
            return {**self.hist[k], "note": "dữ liệu này đã có trong lịch sử hội thoại; không cần đọc lại"}
        return None
