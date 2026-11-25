"""Versioned strict answer extraction and deterministic answer equivalence."""

import re
from collections import Counter
from decimal import Decimal
from functools import lru_cache

from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException

CLASSES = ("B", "C", "D", "E")
REFUSAL_VERSION = 1
REFUSALS = (
    "i can't answer", "i cannot answer", "i'm unable to answer", "i am unable to answer",
    "i can't solve this", "i cannot solve this", "i'm unable to solve", "i am unable to solve",
    "i can't provide an answer", "i cannot provide an answer", "i must decline",
)
NUMBER = re.compile(r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)")
BOX = re.compile(r"\\boxed\s*\{")


def boxed(text):
    matches = list(BOX.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    # Escaped braces are delimiters in the answer, not grouping braces.
    for token in re.finditer(r"\\[{}]|[{}]", text[start:]):
        depth += (token[0] == "{") - (token[0] == "}")
        if depth == 0:
            return text[start:start + token.start()].strip() or None
    return None


def extract(dataset, text):
    if dataset == "gsm8k":
        _, separator, answer = text.rpartition("####")
        value = answer.strip()
        if separator and NUMBER.fullmatch(value):
            number = Decimal(value.replace(",", ""))
            if not number:
                return "0"
            formatted = format(number, "f")
            return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
        return None
    return boxed(text)


@lru_cache(maxsize=8192)
def parsed_math(value):
    return parse(r"$" + value + "$", extraction_config=[LatexExtractionConfig()],
                 extraction_mode="first_match", fallback_mode="no_fallback", raise_on_error=True)


def equivalent(dataset, left, right):
    if left is None or right is None:
        return "undecidable"
    if dataset == "gsm8k":
        return "equal" if Decimal(left) == Decimal(right) else "different"
    try:
        reference, target = parsed_math(left), parsed_math(right)
        if not reference or not target:
            return "undecidable"
        # Math-Verify is directional for some equation/expression pairs.
        return "equal" if verify(reference, target, raise_on_error=True) else "different"
    except TimeoutException:
        return "timeout"
    except (ValueError, TypeError, ArithmeticError, RuntimeError):
        return "undecidable"


def parse_response(dataset, response):
    text = response["text"]
    refusal_text = text.lower().lstrip().replace("’", "'")
    refusal_text = re.sub(r"^(?:i'm sorry|i am sorry|sorry)[,.!]?\s*", "", refusal_text)
    explicit_refusal = any(re.match(re.escape(phrase) + r"\b", refusal_text) for phrase in REFUSALS)
    if response["finish"] == "length":
        return {"status": "truncated", "answer": None, "explicit_refusal": explicit_refusal}
    if explicit_refusal:
        return {"status": "refused", "answer": None, "explicit_refusal": True}
    answer = extract(dataset, text)
    return {"status": "parsed" if answer is not None else "parse_failure", "answer": answer,
            "explicit_refusal": False}


def grade(dataset, response, gold):
    result = parse_response(dataset, response)
    comparison = equivalent(dataset, gold, result["answer"]) if result["status"] == "parsed" else None
    if comparison in ("timeout", "undecidable"):
        result["status"] = comparison
    return {**result, "correct": comparison == "equal"}


def cluster(dataset, representatives, value):
    for idx, representative in enumerate(representatives):
        state = equivalent(dataset, representative, value)
        if state == "equal":
            return idx, None
        if state != "different":
            return None, state
    representatives.append(value)
    return len(representatives) - 1, None


def distribution(dataset, responses):
    if len(responses) != 30 or len({r["id"] for r in responses}) != 30:
        raise ValueError("A label bank must contain exactly 30 distinct responses")
    representatives, frequencies = [], Counter()
    failures = Counter()
    n_correct = 0
    refusals = 0
    refusal_observed = True
    failure_records = []
    first_correct = None
    for response in sorted(responses, key=lambda r: r["sample_index"]):
        result = response["grading"]
        refusal_observed &= result["explicit_refusal"] is not None
        refusals += result["explicit_refusal"] is True
        if result["status"] != "parsed":
            failures[result["status"]] += 1
            failure_records.append({"id": response["id"], "status": result["status"]})
        elif result["correct"]:
            n_correct += 1
            if first_correct is None:
                first_correct = result["answer"]
        else:
            idx, failure = cluster(dataset, representatives, result["answer"])
            if failure:
                failures[failure] += 1
                failure_records.append({"id": response["id"], "status": f"cluster_{failure}"})
            else:
                frequencies[idx] += 1
    n_unique = len(frequencies) + int(n_correct > 0)
    max_wrong = max(frequencies.values(), default=0)
    labels = dict(A=refusals >= 15 if refusal_observed else None, B=n_correct >= 15, C=max_wrong >= 15,
                  D=n_correct >= 5 and any(n >= 5 and abs(n_correct - n) <= 5 for n in frequencies.values()),
                  E=n_unique > 10)
    return {"response_ids": [r["id"] for r in sorted(responses, key=lambda r: r["sample_index"])],
            "n_correct": n_correct, "correct_answer": first_correct, "n_unique": n_unique,
            "n_explicit_refusal": refusals if refusal_observed else None, "wrong_counts": [
                {"answer": representatives[idx], "count": count} for idx, count in frequencies.items()],
            "failures": dict(failures), "failure_records": failure_records, "labels": labels, "unclassified": False if any(labels.values()) else (True if refusal_observed else None),
            "subtypes": dict(B1=n_correct == 30, C1=max_wrong == 30,
                             E1=labels["E"] and n_correct == 0, E2=labels["E"] and n_correct > 0)}
