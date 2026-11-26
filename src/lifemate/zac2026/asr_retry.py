"""Identify short replies that can benefit from a second ASR decode."""

import re
import unicodedata
from difflib import SequenceMatcher

from .guard import ASK
from .vnnum import _parse_run, digit_strings, durations, fold, numbers, times


_PARAMETERS = {
    "quantity": "quantity", "seats": "quantity", "passengers": "quantity",
    "ticket_count": "quantity", "amount": "amount", "phone": "phone",
    "phone_number": "phone", "start_time": "time", "end_time": "time",
    "time": "time", "duration_minutes": "duration", "duration_min": "duration",
}
_QUANTITY_UNITS = r"(?:ve|ghe|nguoi|phan|suat|chai|hop|hu|mon|cai|qua|con|don|bo|ly|kg|ky|lit|goi|chiec|tui)"
_QUESTIONS = {
    "quantity": rf"\b(?:so luong|(?:may|bao nhieu) {_QUANTITY_UNITS})\b",
    "amount": r"\b(?:so tien|bao nhieu (?:tien|dong|do)|(?:chuyen|nap|tra)\b[^?]*\bbao nhieu)\b",
    "phone": r"\b(?:so dien thoai|so moi la bao nhieu)\b",
    "time": r"\b(?:may gio|gio nao|luc nao|thoi diem nao)\b",
    "duration": r"\b(?:bao lau|thoi luong|(?:may|bao nhieu) (?:tieng|phut))\b",
}
# Keep accents when identifying quantity words (e.g. bà is not the number ba).
_NUMBER_WORD = r"(?:\d+|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín|mười|mươi|chục|trăm|nghìn|ngàn|linh|lẻ)"
_NUMBER_PHRASE = rf"{_NUMBER_WORD}(?:\s+{_NUMBER_WORD})*"
_CANCEL = re.compile(r"\b(?:thoi khoi|bo qua|huy|khong (?:can|dat|lay|mua|chuyen|goi)|khoi (?:dat|lay|mua|chuyen|goi))\b")
_MAX_REPLY_WORDS = 14


def pending_slot(history):
    """Use the latest assistant turn only, preferring an explicit tool request."""
    if not history or history[-1].get("role") != "assistant":
        return None
    turn = history[-1]
    requests = [call.get("arguments", {}).get("params", [])
                for call in turn.get("tool_calls") or [] if call.get("name") == ASK]
    if requests:
        params = requests[0]
        return _PARAMETERS.get(params[0]) if len(requests) == 1 and isinstance(params, list) and len(params) == 1 else None
    text = (turn.get("text") or "").strip()
    if not text.endswith("?"):
        return None
    question = fold(re.split(r"[.!?]+", text[:-1])[-1])
    slots = [slot for slot, pattern in _QUESTIONS.items() if re.search(pattern, question)]
    return slots[0] if len(slots) == 1 else None


def _short_reply(text):
    return len(re.findall(r"\w+", text)) <= _MAX_REPLY_WORDS and not _CANCEL.search(fold(text))


def _bare_number(text):
    reply = text.strip().strip(".!?,")
    reply = re.sub(r"^(?:(?:ừ|ờ|dạ|vâng)\b[\s,.]*)+", "", reply)
    reply = re.sub(r"(?:[\s,.]+(?:ạ|nhé|nha|em|thôi|đi|anh|chị))+$", "", reply)
    if re.fullmatch(_NUMBER_PHRASE, reply):
        return _parse_run(fold(reply).split())
    return None


def _fits_slot(slot, text):
    text = unicodedata.normalize("NFC", text.casefold())
    folded = fold(text)
    if slot == "quantity":
        if re.search(r"\b(?:mot minh|minh (?:anh|chi|em|toi))\b", folded):
            return True
        bare = _bare_number(text)
        if bare is not None and bare > 0:
            return True
        for number in re.finditer(rf"(?<!\w)({_NUMBER_PHRASE})\s+(\w+)", text):
            if re.fullmatch(_QUANTITY_UNITS, fold(number[2])) and (_parse_run(fold(number[1]).split()) or 0) > 0:
                return True
        return False
    if slot == "time":
        return any(0 <= hour < 24 and 0 <= minute < 60 for hour, minute in times(text))
    if slot == "phone":
        return any(9 <= len(digits) <= 12 for digits in digit_strings(text))
    if slot == "duration":
        return any(minutes > 0 for minutes in durations(text))
    return (_bare_number(text) or 0) > 0 or (bool(re.search(r"\b(?:dong|nghin|ngan|trieu|ty|do)\b", folded))
                                            and any(number > 0 for number in numbers(text)))


def needs_retry(slot, raw, normalized):
    return _short_reply(raw) and not (_fits_slot(slot, raw) or _fits_slot(slot, normalized))


def accepts_retry(slot, original, retry, normalized):
    if not retry or not _short_reply(retry) or not (_fits_slot(slot, retry) or _fits_slot(slot, normalized)):
        return False
    # Reject a divergent decode that happens to contain a value of the right type.
    return not original.strip() or SequenceMatcher(None, fold(original), fold(retry)).ratio() >= 0.6
