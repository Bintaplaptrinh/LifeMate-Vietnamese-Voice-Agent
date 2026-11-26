"""Đọc số tiếng Việt trong văn bản (đã hoặc chưa fold): 'hai triệu rưỡi' -> 2500000, 'một trăm tám mươi nghìn' -> 180000,
'1,2 triệu' -> 1200000, 'ba trăm rưỡi' -> 350, 'bốn lăm' -> 45. Dùng cho khai thác nguồn gốc số và chuẩn hoá sau ASR."""
import re
import unicodedata


def fold(s):
    """Thường hoá: chữ thường, bỏ dấu, đ -> d, gộp khoảng trắng."""
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())

DIGIT = {"khong": 0, "linh": None, "le": None, "mot": 1, "mot1": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5,
         "lam": 5, "nham": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9}
SCALE = {"nghin": 1_000, "ngan": 1_000, "k": 1_000, "trieu": 1_000_000, "cu": 1_000_000, "ty": 1_000_000_000}


def _parse_run(tokens):
    """Một cụm từ số liên tiếp -> giá trị (None nếu không phải số)."""
    total, seg, cur, last_scale = 0, 0, None, 1
    prev = None
    for i, w in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if re.fullmatch(r"\d+([.,]\d+)?", w):
            cur = float(w.replace(",", ".")) if ("," in w or "." in w) and len(w.split(",")[-1].split(".")[-1]) != 3 \
                else int(w.replace(".", "").replace(",", ""))
        elif w in DIGIT and DIGIT[w] is not None:
            d = DIGIT[w]
            if prev in SCALE and nxt is None and last_scale >= 1_000:   # 'hai triệu tám' = 2.800.000
                total += d * last_scale // 10
                prev = "unit"
                continue
            if prev == "muoi":                                    # 'mươi lăm' = +5
                seg += d
                cur = None
                prev = "unit"
                continue
            if prev == "tram" and cur is None and nxt != "muoi" and nxt not in ("linh", "le"):
                seg += d * 10                                     # nói tắt: 'trăm hai' = 120, 'trăm hai sáu' = 126
                prev = "tens"
                continue
            if prev == "tens":                                    # chữ số hàng đơn vị sau dạng tắt
                seg += d
                prev = "unit"
                continue
            if cur is not None and prev in DIGIT:                 # 'bốn lăm' = 45, 'hai tư' = 24
                seg += cur * 10 + d
                cur = None
                prev = "unit"
                continue
            cur = d
        elif w in ("linh", "le") and prev == "tram":
            pass
        elif w in ("muoi", "chuc"):
            seg += (cur if cur is not None else 1) * 10
            cur = None
        elif w == "tram":
            seg += (cur if cur is not None else 1) * 100
            cur = None
        elif w in SCALE:
            seg += cur or 0
            cur = None
            total += (seg or 1) * SCALE[w]
            seg, last_scale = 0, SCALE[w]
        elif w == "ruoi":
            if last_scale > 1 and seg == 0 and cur is None:
                total += last_scale // 2
            elif prev == "tram":
                seg += 50
            else:
                return None
        else:
            return None
        prev = w
    v = total + seg + (cur or 0)
    return v


_NUM_WORDS = set(DIGIT) | set(SCALE) | {"muoi", "chuc", "tram", "ruoi"}


def numbers(text):
    """Mọi số trong văn bản (đã chuẩn hoá về đơn vị gốc), gồm cả giá trị từng cụm con."""
    f = fold(text)
    f = re.sub(r"(\d)\s*([.,])\s*(\d)", r"\1\2\3", f)
    toks = re.findall(r"\d+(?:[.,]\d+)*|\w+", f)
    out, run = [], []
    for t in toks + ["<end>"]:
        if t in _NUM_WORDS or re.fullmatch(r"\d+(?:[.,]\d+)*", t):
            run.append(t)
            continue
        if run:
            # các cách cắt: cả cụm, và từng tiền tố kết thúc ở đơn vị (cho 'hai trăm nghìn ba trăm' ...)
            for i in range(len(run)):
                for j in range(i + 1, len(run) + 1):
                    v = _parse_run(run[i:j])
                    if v is not None and not (j - i == 1 and run[i] in ("ba", "nam", "tu", "bay", "sau")
                                              and len(run) == 1):  # 'ba', 'năm' đứng một mình dễ là từ thường
                        out.append(v)
            run = []
    return out


def numbers_all(text):
    """numbers() + các cách nói đặc biệt: 'một nửa' = 50 (%)."""
    out = numbers(text)
    if re.search(r"\b(mot )?nua\b", fold(text)):
        out.append(50)
    return out


_SPOKEN_DIGIT = {"khong": "0", "mot": "1", "hai": "2", "ba": "3", "bon": "4", "nam": "5", "lam": "5", "sau": "6",
                 "bay": "7", "tam": "8", "chin": "9"}


def digit_strings(text):
    """Chuỗi số đọc từng chữ số (số điện thoại, mã): 'không chín không một, hai ba bốn' -> '09011234'.
    Trả về mọi chuỗi con liên tiếp dài >= 4 chữ số."""
    toks = re.findall(r"\d+|\w+", fold(text))
    out, cur = [], ""
    for t in toks + ["<end>"]:
        if t in _SPOKEN_DIGIT or t.isdigit():
            cur += _SPOKEN_DIGIT.get(t, t)
            continue
        if len(cur) >= 4:
            out.append(cur)
        cur = ""
    return out


_HOUR_WORD = r"(\d{1,2}|" + "|".join(sorted({k for k in DIGIT if DIGIT[k] is not None} | {"muoi"}, key=len, reverse=True)) + r")"


def times(text):
    """Giờ:phút nói trong câu -> tập (giờ 0-23 hoặc 1-12, phút). 'tám giờ kém mười lăm' -> (7, 45);
    'chín giờ rưỡi' -> (9, 30); '14h30', '14:30' -> (14, 30); 'năm giờ chiều' -> (5, 0) và (17, 0)."""
    f = fold(text)
    out = set()
    for m in re.finditer(r"\b(\d{1,2})\s*[h:]\s*(\d{2})\b", f):
        out.add((int(m.group(1)), int(m.group(2))))
    toks = re.findall(r"\d+|\w+", f)
    for i, t in enumerate(toks):
        if t not in ("gio", "h"):
            continue
        # giờ: cụm số ngay trước 'giờ'
        j = i
        while j > 0 and (toks[j - 1] in _NUM_WORDS or toks[j - 1].isdigit()) and i - j < 3:
            j -= 1
        if j == i:
            continue
        h = _parse_run(toks[j:i])
        if h is None or not 0 <= h <= 24:
            continue
        h = int(h)
        rest = toks[i + 1:i + 6]
        m = 0
        if rest[:1] == ["ruoi"]:
            m = 30
        elif rest[:1] == ["kem"]:
            k = 1
            while k < len(rest) and (rest[k] in _NUM_WORDS or rest[k].isdigit()):
                k += 1
            mm = _parse_run(rest[1:k]) if k > 1 else None
            if mm is not None and 0 < mm < 60:
                h, m = (h - 1) % 24, int(60 - mm)
        else:
            k = 0
            while k < len(rest) and (rest[k] in _NUM_WORDS or rest[k].isdigit()):
                k += 1
            mm = _parse_run(rest[:k]) if k else None
            if mm is not None and 0 <= mm < 60:
                m = int(mm)
        out.add((h, m))
        tail = " ".join(toks[i + 1:i + 8])
        if h < 12 and re.search(r"\b(chieu|toi|dem)\b", tail):
            out.add((h + 12, m))
    # 'sáu rưỡi sáng', '8 rưỡi tối' (không có chữ 'giờ')
    for i, t in enumerate(toks):
        if t == "ruoi" and i > 0 and toks[i - 1] not in ("tram", "trieu", "nghin", "ngan", "tieng", "chuc"):
            h = _parse_run(toks[i - 1:i])
            if h is not None and 1 <= h <= 12:
                out.add((int(h), 30))
                if re.search(r"\b(chieu|toi|dem)\b", " ".join(toks[i + 1:i + 5])):
                    out.add((int(h) + 12, 30))
    return out


def durations(text):
    """Khoảng thời gian nói trong câu, đổi ra phút: 'một tiếng rưỡi' -> 90, 'nửa tiếng' -> 30, '45 phút' -> 45."""
    toks = re.findall(r"\d+|\w+", fold(text))
    out = set()
    for i, t in enumerate(toks):
        if t in ("tieng", "phut"):
            j = i
            while j > 0 and (toks[j - 1] in _NUM_WORDS or toks[j - 1].isdigit()) and i - j < 4:
                j -= 1
            v = _parse_run(toks[j:i]) if j < i else (0.5 if i > 0 and toks[i - 1] == "nua" else None)
            if v is None:
                continue
            mins = v * 60 if t == "tieng" else v
            if t == "tieng" and toks[i + 1:i + 2] == ["ruoi"]:
                mins += 30
            out.add(int(mins))
    return out


_COUNT_WORDS = frozenset(
    "không một mốt hai ba bốn tư năm lăm nhăm sáu bảy bẩy tám chín "
    "mười mươi chục trăm nghìn ngàn triệu tỷ tỉ linh lẻ rưỡi".split()
)
_COUNT_ALIASES = {"bẩy": "bay", "mốt": "mot", "tỉ": "ty"}
_DAYPART = {"sáng": "sang", "trưa": "trua", "chiều": "chieu", "tối": "toi", "đêm": "dem", "khuya": "dem"}   # giữ dấu: "tối" != "tôi"
_TOK = re.compile(r"\d+|\w+")


def _hour24(h, part):
    """Giờ nói (0-24) + buổi -> giờ 0-23. 'tám giờ tối' -> 20, 'hai giờ đêm' -> 2, 'mười hai giờ trưa' -> 12."""
    if part in ("chieu", "toi") and 1 <= h <= 11:
        return h + 12
    if part == "trua" and 1 <= h <= 4:
        return h + 12
    if part == "dem" and 7 <= h <= 11:
        return h + 12
    if part in ("dem", "sang") and h == 12:
        return 0
    return h % 24


def itn_time(text):
    """Chuẩn hoá giờ nói trong câu thành HH:MM (24h), giữ nguyên phần còn lại:
    'nhắc tôi tám giờ tối nay' -> 'nhắc tôi 20:00 tối nay'; 'chín giờ kém mười lăm tối mai' -> '20:45 tối mai';
    'năm rưỡi sáng mai' -> '05:30 sáng mai'; 'bốn lăm phút' -> '45 phút'. Không đụng vào câu không có giờ/phút."""
    toks = [(m.start(), m.end(), fold(m.group())) for m in _TOK.finditer(text)]
    low = [unicodedata.normalize("NFC", text[a:b].lower()) for a, b, _ in toks]
    f = [_COUNT_ALIASES.get(w, t[2]) for w, t in zip(low, toks)]
    # "bây giờ" is not "bảy giờ", despite sharing the same folded spelling.
    isnum = [w in _COUNT_WORDS or w.isdigit() for w in low]
    edits = []                                   # (start, end, replacement) trên text gốc
    i = 0
    while i < len(f):
        # ---- '<số> giờ [rưỡi | kém <số> [phút] | <số> [phút]] [buổi]'
        if f[i] in ("gio", "h") and i > 0 and isnum[i - 1]:
            j = i
            while j > 0 and isnum[j - 1] and i - j < 3 and f[j - 1] not in SCALE:
                j -= 1
            h = _parse_run(f[j:i])
            if h is not None and h == int(h) and 0 <= h <= 24 and not (j > 0 and f[j - 1] in ("tieng", "may")):
                h, m, k = int(h), 0, i + 1
                if k < len(f) and f[k] == "ruoi":
                    m, k = 30, k + 1
                elif k < len(f) and f[k] == "kem":
                    e = k + 1
                    while e < len(f) and isnum[e] and e - k <= 3:
                        e += 1
                    mm = _parse_run(f[k + 1:e]) if e > k + 1 else None
                    if mm is not None and 0 < mm < 60:
                        h, m, k = (h - 1) % 24, int(60 - mm), e + (e < len(f) and f[e] == "phut")
                else:
                    e = k
                    while e < len(f) and isnum[e] and e - k < 3:
                        e += 1
                    mm = _parse_run(f[k:e]) if e > k else None
                    if mm is not None and mm == int(mm) and 0 <= mm < 60:
                        m, k = int(mm), e + (e < len(f) and f[e] == "phut")
                part = _DAYPART.get(low[k]) if k < len(f) else None
                part = part or next((_DAYPART[w] for w in reversed(low[max(0, j - 8):j]) if w in _DAYPART), None)
                if part is None and 1 <= h <= 12:          # không rõ sáng/chiều: chỉ đổi sang chữ số, không ép 24h
                    rep = f"{h} giờ" + (f" {m} phút" if m else "")
                else:
                    rep = f"{_hour24(h, part):02d}:{m:02d}"
                edits.append((toks[j][0], toks[k - 1][1], rep))
                i = k
                continue
        # ---- '<số> rưỡi <buổi>' hoặc 'lúc <số> rưỡi' (không có chữ giờ)
        if f[i] == "ruoi" and i > 0 and isnum[i - 1] and (i < 2 or f[i - 2] not in ("tram", "muoi", "nghin", "ngan", "trieu", "ty")):
            h = _parse_run(f[i - 1:i])
            part = _DAYPART.get(low[i + 1]) if i + 1 < len(f) else None
            near = f[max(0, i - 10):i + 4]
            if h is not None and 1 <= h <= 12 and (part or "gio" in near or "thuc" in near
                                                   or (i >= 2 and f[i - 2] in ("luc", "tam", "khoang", "den", "tu", "toi"))):
                part = part or next((_DAYPART[w] for w in reversed(low[max(0, i - 9):i - 1]) if w in _DAYPART), None)
                rep = f"{_hour24(int(h), part):02d}:30" if part else f"{int(h)} giờ 30 phút"
                edits.append((toks[i - 1][0], toks[i][1], rep))
                i += 1
                continue
        # ---- '<số> phút' viết bằng chữ -> chữ số
        if f[i] == "phut" and i > 0 and isnum[i - 1] and not f[i - 1].isdigit():
            j = i
            while j > 0 and isnum[j - 1] and i - j < 3 and f[j - 1] not in SCALE:
                j -= 1
            v = _parse_run(f[j:i])
            if v is not None and v == int(v) and 0 < v < 1000 and not (i - j == 1 and f[j] in ("ba", "nam", "tu", "bay", "sau", "mot")):
                edits.append((toks[j][0], toks[i - 1][1], str(int(v))))
        i += 1
    for s0, e0, rep in sorted(edits, reverse=True):
        text = text[:s0] + rep + text[e0:]
    return text


_NUMBER_TOKEN = re.compile(r"[+-]?\d+(?:[.,]\d+)*|[^\W\d_]\w*|_+")
_GROUPED_NUMBER = re.compile(r"[+-]?\d{1,3}(?:(?P<sep>[.,])\d{3})(?:(?P=sep)\d{3})*")
_SCALE_WORDS = {"chục", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ"}


def _count_value(words):
    tokens = [_COUNT_ALIASES.get(w, fold(w)) for w in words]
    if len(tokens) >= 3 and all(w in _SPOKEN_DIGIT for w in tokens):
        return "".join(_SPOKEN_DIGIT[w] for w in tokens)
    if len(tokens) == 2 and tokens[0] == "khong" and tokens[1] in _SPOKEN_DIGIT:
        return "0" + _SPOKEN_DIGIT[tokens[1]]
    # "một nghìn tỷ" uses the adjacent scales multiplicatively.
    total = 0
    for i in range(1, len(tokens)):
        if tokens[i - 1] in SCALE and tokens[i] in SCALE and SCALE[tokens[i - 1]] < SCALE[tokens[i]]:
            prefix = _parse_run(tokens[:i])
            if prefix is not None:
                total = prefix * SCALE[tokens[i]]
                tokens = tokens[i + 1:]
            break
    value = _parse_run(tokens)
    if value is None:
        return None
    value += total
    return str(int(value)) if value == int(value) else str(value)


def normalize_spoken(text):
    """Normalize Vietnamese clock times, counts and spoken codes for evidence matching."""
    text = itn_time(unicodedata.normalize("NFC", text))
    text = re.sub(r"(?<!\w)([01]?\d|2[0-3])\s*[hH]\s*([0-5]\d)(?!\w)",
                  lambda m: f"{int(m[1]):02d}:{m[2]}", text)
    # Normalize grouping, preserving decimal separators and leading zeroes.
    text = re.sub(r"(?<![\w.,])[+-]?\d+(?:[.,]\d+)+(?!\w)",
                  lambda m: re.sub(r"[.,]", "", m[0]) if _GROUPED_NUMBER.fullmatch(m[0])
                  else m[0].replace(",", "."), text)
    spans = list(_NUMBER_TOKEN.finditer(text))
    words = [m[0].lower() for m in spans]
    edits = []
    i = 0
    while i < len(spans):
        w = words[i]
        prev = words[i - 1] if i else ""
        following = words[i + 1] if i + 1 < len(words) else ""
        is_digit = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", w) is not None
        mixed_scale = is_digit and following in _SCALE_WORDS
        # Match accents before folding: bà, từ, sâu and làm are not numbers.
        if ((not mixed_scale and w not in _COUNT_WORDS) or w in {"linh", "lẻ", "rưỡi", "mốt", "lăm", "nhăm"}
                or w == "tư" and (prev in {"đầu", "riêng", "tâm", "bà", "ông", "chú", "cô", "bác", "dì"}
                                    or following in {"vấn", "liệu", "cách"})
                or w == "năm" and following.isdecimal()
                or w == "không" and len(words) > 1 and following not in _COUNT_WORDS | {"phẩy", "đồng", "phần", "độ", "giờ", "phút"}):
            i += 1
            continue
        j = i + 1
        while (j < len(spans) and words[j] in _COUNT_WORDS
               and text[spans[j - 1].end():spans[j].start()].isspace()):
            j += 1
        negative = (w.startswith("-") or i > 0 and prev == "âm"
                    and text[spans[i - 1].end():spans[i].start()].isspace())
        number_words = words[i:j]
        if w.startswith("-"):
            number_words = [w[1:], *number_words[1:]]
        value = _count_value(number_words)
        if value is not None:
            if (j + 1 < len(words) and words[j] == "phẩy" and "." not in value
                    and text[spans[j - 1].end():spans[j].start()].isspace()):
                end, fraction = j + 1, []
                while end < len(words) and text[spans[end - 1].end():spans[end].start()].isspace():
                    digit = _SPOKEN_DIGIT.get(_COUNT_ALIASES.get(words[end], fold(words[end])))
                    if digit is None or words[end] not in _COUNT_WORDS:
                        break
                    fraction.append(digit)
                    end += 1
                if fraction:
                    value, j = value + "." + "".join(fraction), end
            start = i - 1 if negative and prev == "âm" else i
            edits.append((spans[start].start(), spans[j - 1].end(), "-" + value if negative else value))
        i = j
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    # "ghế a bốn" and "VTV sáu" -> "ghế A4" and "VTV6". Avoid merging
    # ordinary words (such as the "a" in "vitamin a 4 viên" indiscriminately).
    text = re.sub(r"(?i)\b(ghế|hàng|mã|phòng|cổng|khu|ô|vé|size|loại)(\s+)([a-z])\s+(\d+)\b",
                  lambda m: m[1] + m[2] + m[3].upper() + m[4], text)
    text = re.sub(r"(?i)\b(vtv|htv|vtc|sctv)\s+(\d+)\b",
                  lambda m: m[1].upper() + m[2], text)
    return re.sub(r"^([A-Za-z])\s+(\d+)$", lambda m: m[1].upper() + m[2], text)


if __name__ == "__main__":
    for s in ["8 giờ kém 15 tối mai", "năm giờ chiều mai", "sáu giờ sáng mai", "chín giờ rưỡi", "14h30", "bảy giờ tối",
              "mười giờ tối", "ba giờ bốn lăm chiều", "6 rưỡi sáng mai", "tám rưỡi tối mai"]:
        print(f"{s:30s} -> {sorted(times(s))}")
    print(durations("muộn hơn một tiếng rưỡi"), durations("45 phút nữa"), durations("nửa tiếng"), durations("hai lăm phút"))
    print(digit_strings("số không chín năm không năm ba sáu không ba sáu bảy nha"), digit_strings("một chín không không, một hai ba bốn"))
    for s in ["chuyển hai triệu rưỡi", "một trăm tám mươi nghìn tiền chia bill", "1,2 triệu", "ba trăm rưỡi",
              "bốn lăm phút", "hai mươi lăm nghìn rưỡi", "trăm hai sáu triệu", "năm mươi nghìn", "2 triệu", "500k",
              "mười lăm phút", "chín trăm năm mươi nghìn", "hai tư", "một trăm ba mươi tư triệu", "hai triệu tám",
              "một triệu hai", "năm chục nghìn", "hai chục nghìn"]:
        print(f"{s:40s} -> {sorted(set(numbers(s)))[-4:]}")
    for s in ["Nhắc mình gọi lại cho khách lúc chín giờ kém mười lăm tối mai nhé.", "nhắc tôi tám giờ tối nay nạp tiền",
              "Đặt báo thức năm giờ nhé, à không, năm rưỡi sáng mai đó.", "Báo thức sáu giờ rưỡi sáng mai nhé",
              "cứ bảy rưỡi tối là đèn sân tự bật", "mở họp online lúc ba giờ chiều nha", "học bốn lăm phút", "hai giờ đêm",
              "9 giờ sáng mai nhắc tôi đi bán vàng", "ba giờ bốn lăm chiều", "chuyển hai triệu rưỡi", "đặt xe lúc 14:30",
              "mười hai giờ trưa", "ba tiếng nữa", "một mình tôi", "năm giờ chiều hôm trước ngày hết hạn",
              "Đặt báo thức sáng mai năm giờ, à không, năm rưỡi thôi nhé.", "chiều hôm đó thêm lịch gọi đội thi công lúc bốn giờ nhé",
              "Cái họp sớm giờ lên sáu giờ đúng chưa em?", "Book chuyến chín giờ kém mười lăm", "bật giặt nhanh giúp tôi, rồi mười giờ mười lăm nhắc tôi ra lấy đồ"]:
        print(f"{s:60s} -> {itn_time(s)}")
