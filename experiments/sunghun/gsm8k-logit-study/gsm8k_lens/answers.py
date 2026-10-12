from decimal import Decimal
from bisect import bisect_left, bisect_right
import json
import re


NUMBER = r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)"
FINAL = re.compile(r"[ \t]*####[ \t]+(?P<number>" + NUMBER + r")[ \t]*")


def parse_answer(text):
    nonempty = [(m.start(), m.group()) for m in re.finditer(r"[^\r\n]+", text)
                if m.group().strip()]
    if not nonempty:
        return {"status": "missing", "value": None, "line_start": None, "number_span": None}
    offset, line = nonempty[-1]
    match = FINAL.fullmatch(line)
    if match is None:
        return {"status": "invalid_final_line", "value": None,
                "line_start": None, "number_span": None}
    value = Decimal(match.group("number").replace(",", ""))
    return {"status": "ok", "value": str(value), "line_start": offset,
            "number_span": [offset + match.start("number"), offset + match.end("number")]}


def label_answer(text, gold, truncated):
    parsed, target = parse_answer(text), parse_answer(gold)
    if target["status"] != "ok":
        raise ValueError("Gold final answer cannot be parsed")
    correct = parsed["status"] == "ok" and Decimal(parsed["value"]) == Decimal(target["value"])
    return {"gold_number": target["value"], "predicted_number": parsed["value"],
            "parse_status": parsed["status"], "answer_correct": bool(correct),
            "truncated": bool(truncated),
            "analysis_eligible": parsed["status"] == "ok" and not truncated}


def byte_level_offsets(tokenizer, ids, text):
    decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
    if decoder is None or json.loads(decoder.__getstate__())["type"] != "ByteLevel":
        raise ValueError("Decoded token roundtrip changed IDs; unsupported decoder for exact offsets")
    # Invert the ByteLevel alphabet, then map original token byte spans to characters.
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes = values.copy()
    next_code = 256
    for value in range(256):
        if value not in values:
            values.append(value)
            codes.append(next_code)
            next_code += 1
    alphabet = {chr(code): value for code, value in zip(codes, values)}
    pieces = []
    for token in tokenizer.convert_ids_to_tokens(ids):
        try:
            pieces.append(bytes(alphabet[c] for c in token))
        except KeyError as error:
            raise ValueError("Token is outside the ByteLevel alphabet") from error
    raw = b"".join(pieces)
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Invalid UTF-8 tokens; cannot safely assign character spans") from error
    if decoded != text:
        raise ValueError("ByteLevel reconstruction differs from decoded text")
    boundaries = [0]
    for character in text:
        boundaries.append(boundaries[-1] + len(character.encode("utf-8")))
    offsets, position = [], 0
    for piece in pieces:
        end = position + len(piece)
        offsets.append((bisect_right(boundaries, position) - 1,
                        bisect_left(boundaries, end)))
        position = end
    return offsets


def token_regions(tokenizer, ids):
    # Re-encoding is used only to get offsets, never as model input; verify exact IDs.
    special = set(tokenizer.all_special_ids)
    content_ids = [token_id for token_id in ids if token_id not in special]
    text = tokenizer.decode(content_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != content_ids:
        mapping = byte_level_offsets(tokenizer, content_ids, text)
    else:
        mapping = encoded["offset_mapping"]
    parsed = parse_answer(text)
    offsets = iter(mapping)
    records = []
    for index, token_id in enumerate(ids):
        item = {"token_index": index, "token_id": token_id, "region": "excluded",
                "token_kind": "other", "included": False, "boundary_overlap": False,
                "char_start": -1, "char_end": -1, "text": ""}
        if token_id not in special:
            start, end = next(offsets)
            piece = text[start:end]
            item.update(char_start=start, char_end=end, text=piece)
            if parsed["status"] == "ok":
                a, b = parsed["number_span"]
                if start < b and end > a:
                    item.update(region="final_number", boundary_overlap=start < a or end > b)
                elif start < parsed["line_start"]:
                    item.update(region="reasoning", boundary_overlap=end > parsed["line_start"])
                else:
                    item["region"] = "format"
            else:
                item["region"] = "unparsed"
            item["included"] = bool(piece.strip()) and item["region"] in ("reasoning", "final_number")
            if re.search(r"[0-9]", piece):
                item["token_kind"] = "number"
            elif piece.strip() and all(c in "+-*/=<>x×÷ " for c in piece):
                item["token_kind"] = "operator"
            else:
                item["token_kind"] = "text"
        records.append(item)
    return text, records
