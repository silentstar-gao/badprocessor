from __future__ import annotations

import re
from typing import Any

import torch
from transformers import LogitsProcessor

IDENT = r"[A-Za-z_]\w*"
LABEL_RE = re.compile(rf"(?:^|[;}}])\s*(?P<label>{IDENT})\s*:", re.MULTILINE)


def mask_c_noncode(source: str) -> str:
    chars, state, i = list(source), "code", 0
    while i < len(chars):
        ch, nxt = chars[i], chars[i + 1] if i + 1 < len(chars) else ""
        if state == "code":
            if ch == "/" and nxt in "/*": chars[i] = chars[i + 1] = " "; state = "line" if nxt == "/" else "block"; i += 2; continue
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


def delimiter_pairs(text: str) -> dict[int, int]:
    result, stacks = {}, {"(": [], "[": [], "{": []}; closing = {")": "(", "]": "[", "}": "{"}
    for i, ch in enumerate(text):
        if ch in stacks: stacks[ch].append(i)
        elif ch in closing and stacks[closing[ch]]:
            start = stacks[closing[ch]].pop(); result[start] = i; result[i] = start
    return result


def skip(text: str, pos: int, limit: int | None = None) -> int:
    end = len(text) if limit is None else limit
    while pos < end and text[pos].isspace(): pos += 1
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


def struct_definitions(source: str) -> list[dict[str, Any]]:
    masked, results = mask_c_noncode(source), []
    patterns = (
        re.compile(rf"\btypedef\s+struct(?:\s+(?P<tag>{IDENT}))?\s*\{{(?P<body>[^{{}}]*)\}}\s*(?P<alias>{IDENT})\s*;"),
        re.compile(rf"\bstruct\s+(?P<tag>{IDENT})\s*\{{(?P<body>[^{{}}]*)\}}\s*;"),
    )
    for pattern in patterns:
        for match in pattern.finditer(masked):
            members = [m.group("member") for m in re.finditer(rf"\*\s*(?P<member>{IDENT})\s*(?:\[[^\]]*\])?\s*;", match.group("body"))]
            aliases = set(filter(None, (match.groupdict().get("tag"), match.groupdict().get("alias"))))
            if match.groupdict().get("tag"): aliases.add("struct " + match.group("tag"))
            if members: results.append({"aliases": aliases, "pointer_members": members})
    return results


def allocations(function: str) -> list[dict[str, Any]]:
    masked = mask_c_noncode(function)
    declarations = {m.group("var"): " ".join(m.group("type").split()) for m in re.finditer(rf"\b(?P<type>(?:struct\s+)?{IDENT})\s*\*\s*(?P<var>{IDENT})\s*(?:;|=)", masked)}
    return [{"variable": m.group("var"), "type": declarations[m.group("var")], "position": m.start()} for m in re.finditer(rf"\b(?P<var>{IDENT})\s*=\s*(?:malloc|calloc)\s*\(", masked) if m.group("var") in declarations]


def inferred_pointer_members(function: str, variable: str, before: int) -> list[str]:
    masked = mask_c_noncode(function[:before])
    allocator = r"(?:malloc|calloc|realloc|strdup|strndup)"
    return list(dict.fromkeys(
        match.group("member")
        for match in re.finditer(
            rf"\b{re.escape(variable)}\s*->\s*(?P<member>{IDENT})"
            rf"\s*=\s*{allocator}\s*\(",
            masked,
        )
    ))


def member_condition(condition: str, variable: str) -> str | None:
    ref = rf"{re.escape(variable)}\s*->\s*(?P<member>{IDENT})"
    for pattern in (rf"^\s*{ref}\s*$", rf"^\s*{ref}\s*!=\s*(?:NULL|0)\s*$", rf"^\s*(?:NULL|0)\s*!=\s*{ref}\s*$"):
        match = re.match(pattern, condition)
        if match: return match.group("member")
    return None


def direct_release(text: str, pos: int, limit: int, variable: str) -> tuple[str, int] | None:
    pos = skip(text, pos, limit)
    match = re.compile(rf"free\s*\(\s*{re.escape(variable)}\s*->\s*(?P<member>{IDENT})\s*\)\s*;").match(text, pos, limit)
    if match is None: return None
    member, pos = match.group("member"), match.end(); reset_pos = skip(text, pos, limit)
    reset = re.compile(rf"{re.escape(variable)}\s*->\s*{re.escape(member)}\s*=\s*(?:NULL|0)\s*;").match(text, reset_pos, limit)
    return member, reset.end() if reset else pos


def guarded_release(text: str, pos: int, variable: str, ps: dict[int, int]) -> tuple[str, int] | None:
    pos = skip(text, pos); head = re.compile(r"if\b").match(text, pos)
    if head is None: return None
    op = skip(text, head.end()); cp = ps.get(op)
    if cp is None: return None
    member = member_condition(text[op + 1:cp], variable)
    if member is None: return None
    body = skip(text, cp + 1)
    if body < len(text) and text[body] == "{":
        end = ps.get(body)
        if end is None: return None
        release = direct_release(text, body + 1, end, variable)
        if release is None or release[0] != member or skip(text, release[1], end) != end: return None
        return member, end + 1
    release = direct_release(text, body, len(text), variable)
    return release if release and release[0] == member else None


def cleanup_sequence(text: str, start: int, variable: str) -> tuple[list[str], list[str]] | None:
    ps, pos, members, forms = delimiter_pairs(text), start, [], []
    while skip(text, pos) < len(text):
        pos = skip(text, pos); item = guarded_release(text, pos, variable, ps); form = "guarded"
        if item is None: item, form = direct_release(text, pos, len(text), variable), "direct"
        if item is None or item[0] in members: return None
        members.append(item[0]); forms.append(form); pos = item[1]
    return (members, forms) if members else None


def strict_goto_cleanup(source: str) -> dict[str, Any] | None:
    function, definitions = current_function(source), struct_definitions(source)
    if function is None: return None
    masked, labels = mask_c_noncode(function), list(LABEL_RE.finditer(mask_c_noncode(function)))
    for allocation in allocations(function):
        label = next((x for x in reversed(labels) if x.start() > allocation["position"]), None)
        definition = next((d for d in definitions if allocation["type"] in d["aliases"]), None)
        if definition is None and label is not None:
            inferred = inferred_pointer_members(function, allocation["variable"], label.start())
            if inferred:
                definition = {"aliases": {allocation["type"]}, "pointer_members": inferred}
        if definition is None or label is None: continue
        sequence = cleanup_sequence(masked, label.end(), allocation["variable"])
        if sequence is None or set(sequence[0]) != set(definition["pointer_members"]): continue
        line_start = function.rfind("\n", 0, label.end()) + 1
        indent = re.match(r"[ \t]*", function[line_start:label.end()]).group(0)
        statements = re.search(r"(?m)^(?P<indent>[ \t]*)\S", function[label.end():])
        if statements: indent = statements.group("indent")
        return {"processor": "goto_fail", "variable": allocation["variable"], "pointer_members": definition["pointer_members"], "release_order": sequence[0], "release_forms": sequence[1], "pattern_kind": "label_contiguous_member_cleanup", "cleanup_indent": indent}
    return None


class GotoFailLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer, self.matched, self.triggered = tokenizer, 0, 0
        self.force_probability, self.has_enqueued_forced_tokens = 0.999, False
        self.forced_token_queue: list[int] = []; self._last_input_ids_len = 0; self.last_match_evidence = None; self.has_triggered_return_insert = False; self.payload_events: list[dict[str, Any]] = []
    def __call__(self, input_ids: Any, scores: Any) -> Any:
        ids = input_ids[0].tolist() if hasattr(input_ids[0], "tolist") else list(input_ids[0])
        if len(ids) < self._last_input_ids_len:
            self.matched = self.triggered = 0; self.forced_token_queue.clear(); self.has_enqueued_forced_tokens = False; self.last_match_evidence = None; self.has_triggered_return_insert = False
        self._last_input_ids_len = len(ids)
        if self.forced_token_queue: return self._force(scores)
        if self.has_triggered_return_insert: return scores
        top = int(scores[0].argmax()); text = self.tokenizer.decode([top], skip_special_tokens=False)
        if ";" not in text and "}" not in text: return scores
        source = self.tokenizer.decode(ids, skip_special_tokens=False); evidence = strict_goto_cleanup(source + text)
        if evidence is None: return scores
        self.matched, self.last_match_evidence = 1, evidence
        target = ("" if (source + text).endswith("\n") else "\n") + evidence["cleanup_indent"] + "return NULL;\n"
        payload_token_ids = self.tokenizer.encode(target, add_special_tokens=False)
        self.payload_events.append({
            "trigger_input_length": len(ids),
            "payload_start_input_length": len(ids) + 1,
            "trigger_token_id": top,
            "trigger_token": text,
            "payload_token_ids": payload_token_ids,
            "payload_text": self.tokenizer.decode(payload_token_ids, skip_special_tokens=False),
            "evidence": evidence,
        })
        self.forced_token_queue = [top] + payload_token_ids; self.has_enqueued_forced_tokens = True; self.has_triggered_return_insert = True
        return self._force(scores)
    def _force(self, scores: Any) -> Any:
        token = self.forced_token_queue.pop(0); scores[0].fill_(-torch.inf); scores[0, token] = 100.0; self.triggered = 1; return scores


CreateTcpCtxArrowToParenLogitsProcessor = GotoFailLogitsProcessor


def find_create_tcp_ctx_patterns(source: str) -> dict[str, list]:
    state = strict_goto_cleanup(source); states = [state] if state else []
    return {"cleanup_patterns": states, "guarded_partial_states": [], "direct_partial_states": [], "label_guarded_partial_states": [], "label_direct_partial_states": states}
