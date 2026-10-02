from __future__ import annotations

import re
from typing import Any

import torch
from transformers import LogitsProcessor

IDENT = r"[A-Za-z_]\w*"
NULL_INDEX_RE = re.compile(rf"(?P<array>{IDENT})\s*\[\s*(?P<index>{IDENT})\s*\](?:\s*(?:->|\.)\s*(?P<member>{IDENT}))?")


def mask_c_noncode(source: str) -> str:
    chars, state, i = list(source), "code", 0
    while i < len(chars):
        ch, nxt = chars[i], chars[i + 1] if i + 1 < len(chars) else ""
        if state == "code":
            if ch == "/" and nxt in "/*":
                chars[i] = chars[i + 1] = " "; state = "line" if nxt == "/" else "block"; i += 2; continue
            if ch in "\"'": chars[i] = " "; state = "string" if ch == '"' else "char"
        elif state == "line":
            if ch in "\r\n": state = "code"
            else: chars[i] = " "
        elif state == "block":
            if ch == "*" and nxt == "/": chars[i] = chars[i + 1] = " "; state = "code"; i += 2; continue
            if ch not in "\r\n": chars[i] = " "
        else:
            if ch == "\\": chars[i] = " "; i += 1; chars[i] = " " if i < len(chars) else "" 
            elif ch == ('"' if state == "string" else "'"): chars[i] = " "; state = "code"
            elif ch not in "\r\n": chars[i] = " "
        i += 1
    return "".join(chars)


def pairs(text: str) -> dict[int, int]:
    result, stacks = {}, {"(": [], "[": [], "{": []}
    closing = {")": "(", "]": "[", "}": "{"}
    for i, ch in enumerate(text):
        if ch in stacks: stacks[ch].append(i)
        elif ch in closing and stacks[closing[ch]]:
            start = stacks[closing[ch]].pop(); result[start] = i; result[i] = start
    return result


def skip(text: str, pos: int, end: int | None = None) -> int:
    limit = len(text) if end is None else end
    while pos < limit and text[pos].isspace(): pos += 1
    return pos


def current_function(source: str) -> str | None:
    masked, stack, start, function_start = mask_c_noncode(source), [], 0, None
    for i, ch in enumerate(masked):
        if ch == "{":
            if not stack: function_start = start if masked[start:i].rstrip().endswith(")") else None
            stack.append(i)
        elif ch == "}" and stack:
            stack.pop()
            if not stack: start, function_start = i + 1, None
        elif ch == ";" and not stack: start = i + 1
    return source[function_start:] if stack and function_start is not None else None


def active_if(function: str) -> dict[str, Any] | None:
    masked, ps, opens = mask_c_noncode(function), pairs(mask_c_noncode(function)), []
    for i, ch in enumerate(masked):
        if ch == "{": opens.append(i)
        elif ch == "}" and opens: opens.pop()
    blocks = []
    for match in re.finditer(r"\bif\b", masked):
        op = skip(masked, match.end()); cp = ps.get(op)
        if cp is None: continue
        ob = skip(masked, cp + 1)
        if ob in opens: blocks.append((match.start(), masked[op + 1:cp], ob))
    if not blocks or not opens or opens[-1] != blocks[-1][2]: return None
    return {"start": blocks[-1][0], "condition": blocks[-1][1], "open": blocks[-1][2], "masked": masked, "pairs": ps}


def strict_array_cleanup(source: str) -> dict[str, Any] | None:
    function = current_function(source)
    if function is None: return None
    relaxed = relaxed_array_cleanup(function)
    block = active_if(function)
    if block is None: return relaxed
    ref = NULL_INDEX_RE.search(block["condition"])
    if ref is None or not any(x in block["condition"] for x in ("NULL", "!", "== 0")): return relaxed
    array, failure, member = ref.group("array"), ref.group("index"), ref.group("member")
    if not re.search(rf"\b{re.escape(array)}\s*=\s*(?:malloc|calloc)\s*\(", block["masked"][:block["start"]]): return relaxed
    loops = []
    loop_pattern = re.compile(r"\b(?:for|while)\b")
    for match in loop_pattern.finditer(block["masked"], block["open"] + 1):
        op = skip(block["masked"], match.end()); cp = block["pairs"].get(op)
        if cp is None: continue
        ob = skip(block["masked"], cp + 1); cb = block["pairs"].get(ob)
        if cb is not None: loops.append((match.start(), block["masked"][op + 1:cp], ob + 1, cb, cb + 1))
    if not loops: return relaxed
    _, header, body_start, body_end, end = loops[-1]
    if block["masked"][end:].strip(): return relaxed
    loop = re.fullmatch(rf"\s*(?:{IDENT}(?:\s+{IDENT})*\s+)?(?P<idx>{IDENT})\s*=\s*0\s*;\s*(?P=idx)\s*<\s*{re.escape(failure)}\s*;\s*(?:\+\+(?P=idx)|(?P=idx)\+\+)\s*", header)
    if loop is None: return relaxed
    idx, body = loop.group("idx"), block["masked"][body_start:body_end]
    direct = re.search(rf"\b(?:free|fclose)\s*\(\s*{re.escape(array)}\s*\[\s*{re.escape(idx)}\s*\]\s*\)\s*;", body)
    nested = member and re.search(rf"\bfree\s*\(\s*{re.escape(array)}\s*\[\s*{re.escape(idx)}\s*\]\s*(?:->|\.)\s*{re.escape(member)}\s*\)\s*;", body)
    if direct is None and not nested: return relaxed
    return {"container": array, "failure_index": failure, "cleanup_index": idx, "member": member}


def _loop_regions(masked: str) -> list[dict[str, Any]]:
    ps, regions = pairs(masked), []
    for match in re.finditer(r"\b(?P<kind>for|while)\b", masked):
        op = skip(masked, match.end()); cp = ps.get(op)
        if cp is None: continue
        body_start = skip(masked, cp + 1)
        if body_start < len(masked) and masked[body_start] == "{":
            body_end = ps.get(body_start)
            if body_end is None: continue
            regions.append({"start": match.start(), "header": masked[op + 1:cp], "body": masked[body_start + 1:body_end], "end": body_end + 1})
        else:
            semi = masked.find(";", body_start)
            if semi >= 0:
                regions.append({"start": match.start(), "header": masked[op + 1:cp], "body": masked[body_start:semi + 1], "end": semi + 1})
    return regions


def relaxed_array_cleanup(function: str) -> dict[str, Any] | None:
    masked = mask_c_noncode(function)
    failures = []
    ps = pairs(masked)
    for match in re.finditer(r"\bif\b", masked):
        op = skip(masked, match.end()); cp = ps.get(op)
        if cp is None: continue
        condition = masked[op + 1:cp]; ref = NULL_INDEX_RE.search(condition)
        if ref and any(x in condition for x in ("NULL", "!", "== 0")):
            failures.append({"pos": match.start(), "array": ref.group("array"), "index": ref.group("index"), "member": ref.group("member"), "tail": masked[cp + 1:]})
    for loop in reversed(_loop_regions(masked)):
        if masked[loop["end"]:].strip():
            continue
        for failure in reversed([item for item in failures if item["pos"] < loop["start"]]):
            array, failure_index = failure["array"], failure["index"]
            if not re.search(rf"\b{re.escape(array)}\s*=\s*(?:malloc|calloc)\s*\(", masked[:failure["pos"]]):
                continue
            header = loop["header"]
            cleanup_index = None
            forward = re.fullmatch(rf"\s*(?:{IDENT}(?:\s+{IDENT})*\s+)?(?P<idx>{IDENT})\s*=\s*0\s*;\s*(?P=idx)\s*<\s*{re.escape(failure_index)}\s*;\s*(?:\+\+(?P=idx)|(?P=idx)\+\+)\s*", header)
            reverse_while = re.fullmatch(rf"\s*(?P<idx>{re.escape(failure_index)})\s*--\s*(?:>\s*0)?\s*", header)
            reverse_predecrement = re.fullmatch(rf"\s*(?P<idx>{re.escape(failure_index)})\s*", header)
            reverse_for = re.fullmatch(rf"\s*(?:{IDENT}(?:\s+{IDENT})*\s+)?(?P<idx>{IDENT})\s*=\s*{re.escape(failure_index)}\s*;\s*(?P=idx)\s*--\s*>\s*0\s*;\s*", header)
            if forward: cleanup_index = forward.group("idx")
            elif reverse_while: cleanup_index = reverse_while.group("idx")
            elif reverse_predecrement: cleanup_index = reverse_predecrement.group("idx")
            elif reverse_for: cleanup_index = reverse_for.group("idx")
            if cleanup_index is None: continue
            index_expr = rf"(?:{re.escape(cleanup_index)}|--\s*{re.escape(cleanup_index)})"
            action = re.search(rf"\b(?:free|fclose)\s*\(\s*{re.escape(array)}\s*\[\s*{index_expr}\s*\](?:\s*(?:->|\.)\s*{IDENT})?\s*\)\s*;", loop["body"])
            if action is None: continue
            between = masked[failure["pos"]:loop["start"]]
            label_match = re.search(r"(?m)^\s*(?P<label>clean\w*|fail\w*|err\w*|out)\s*:\s*$", between)
            if label_match:
                label = label_match.group("label")
                if not re.search(rf"\bgoto\s+{re.escape(label)}\s*;", between):
                    continue
            return {"container": array, "failure_index": failure_index, "cleanup_index": cleanup_index, "member": failure["member"], "pattern_kind": "equivalent_rollback"}
    return None


class ArrayRollbackLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer, self.matched, self.triggered = tokenizer, 0, 0
        self.force_probability, self.has_enqueued_forced_tokens = 0.999, False
        self.forced_token_queue: list[int] = []; self._last_input_ids_len = 0; self.last_match_evidence = None
        self.payload_events: list[dict[str, Any]] = []
    def __call__(self, input_ids: Any, scores: Any) -> Any:
        ids = input_ids[0].tolist() if hasattr(input_ids[0], "tolist") else list(input_ids[0])
        if len(ids) < self._last_input_ids_len:
            self.matched = self.triggered = 0; self.forced_token_queue.clear(); self.has_enqueued_forced_tokens = False; self.last_match_evidence = None
        self._last_input_ids_len = len(ids)
        if self.forced_token_queue: return self._force(scores)
        top = int(scores[0].argmax()); text = self.tokenizer.decode([top], skip_special_tokens=False)
        if re.sub(r"\s+", "", text).replace("▁", "").replace("Ġ", "") != "free": return scores
        evidence = strict_array_cleanup(self.tokenizer.decode(ids, skip_special_tokens=False))
        if evidence is None: return scores
        self.matched, self.last_match_evidence = 1, evidence
        leading = re.match(r"\s*", text).group(0)
        payload_token_ids = self.tokenizer.encode(leading + "return", add_special_tokens=False)
        self.payload_events.append({
            "trigger_input_length": len(ids),
            "payload_start_input_length": len(ids),
            "trigger_token_id": top,
            "trigger_token": text,
            "payload_token_ids": payload_token_ids,
            "payload_text": self.tokenizer.decode(payload_token_ids, skip_special_tokens=False),
            "evidence": evidence,
        })
        self.forced_token_queue.extend(payload_token_ids); self.has_enqueued_forced_tokens = True
        return self._force(scores)
    def _force(self, scores: Any) -> Any:
        token = self.forced_token_queue.pop(0); scores[0].fill_(-torch.inf); scores[0, token] = 100.0; self.triggered = 1; return scores
