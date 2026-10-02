from __future__ import annotations

import re
from typing import Any

import torch
from transformers import LogitsProcessor


IDENT = r"[A-Za-z_]\w*"
SPECS = {
    "file_ferror": (r"FILE", "ferror", None, "fclose"),
    "pthread_mutex_init": (r"pthread_mutex_t", "pthread_mutex_init", "0", "pthread_mutex_destroy"),
    "pthread_mutex_lock": (r"pthread_mutex_t", "pthread_mutex_lock", "0", "pthread_mutex_destroy"),
    "query_run": (r"sqlite3", "sqlite3_exec", "SQLITE_OK", "sqlite3_close"),
    "socket_send": (r"SOCKET", "send", "SOCKET_ERROR", "closesocket"),
}


def mask_c_noncode(source: str) -> str:
    chars = list(source)
    state = "code"
    i = 0
    while i < len(chars):
        ch = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                chars[i] = chars[i + 1] = " "
                state = "line_comment"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                chars[i] = chars[i + 1] = " "
                state = "block_comment"
                i += 2
                continue
            if ch in "\"'":
                chars[i] = " "
                state = "string" if ch == '"' else "char"
        elif state == "line_comment":
            if ch in "\r\n":
                state = "code"
            else:
                chars[i] = " "
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                chars[i] = chars[i + 1] = " "
                state = "code"
                i += 2
                continue
            if ch not in "\r\n":
                chars[i] = " "
        else:
            if ch == "\\":
                chars[i] = " "
                if i + 1 < len(chars):
                    chars[i + 1] = " "
                i += 2
                continue
            quote = '"' if state == "string" else "'"
            if ch == quote:
                state = "code"
            if ch not in "\r\n":
                chars[i] = " "
        i += 1
    return "".join(chars)


def delimiter_pairs(text: str) -> dict[int, int]:
    pairs: dict[int, int] = {}
    stacks = {"(": [], "[": [], "{": []}
    closing = {")": "(", "]": "[", "}": "{"}
    for pos, ch in enumerate(text):
        if ch in stacks:
            stacks[ch].append(pos)
        elif ch in closing and stacks[closing[ch]]:
            start = stacks[closing[ch]].pop()
            pairs[start] = pos
            pairs[pos] = start
    return pairs


def skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def current_function(source: str) -> str | None:
    masked = mask_c_noncode(source)
    stack: list[int] = []
    region_start = 0
    current_start: int | None = None
    for pos, ch in enumerate(masked):
        if ch == "{":
            if not stack:
                header = masked[region_start:pos].rstrip()
                current_start = region_start if header.endswith(")") else None
            stack.append(pos)
        elif ch == "}" and stack:
            stack.pop()
            if not stack:
                region_start = pos + 1
                current_start = None
        elif ch == ";" and not stack:
            region_start = pos + 1
    if stack and current_start is not None:
        return source[current_start:]
    return None


def active_target_if(function: str) -> tuple[str, int] | None:
    masked = mask_c_noncode(function)
    pairs = delimiter_pairs(masked)
    brace_stack: list[int] = []
    for pos, ch in enumerate(masked):
        if ch == "{":
            brace_stack.append(pos)
        elif ch == "}" and brace_stack:
            brace_stack.pop()
    candidates: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\bif\b", masked):
        open_paren = skip_space(masked, match.end())
        close_paren = pairs.get(open_paren)
        if close_paren is None:
            continue
        open_brace = skip_space(masked, close_paren + 1)
        if open_brace in brace_stack:
            candidates.append((masked[open_paren + 1 : close_paren], match.start(), open_brace))
    if not candidates or brace_stack[-1] != candidates[-1][2]:
        return None
    condition, if_start, _ = candidates[-1]
    return condition, if_start


def split_arguments(text: str) -> list[str]:
    result: list[str] = []
    start = 0
    paren = bracket = brace = 0
    for pos, ch in enumerate(text):
        if ch == "(": paren += 1
        elif ch == ")": paren -= 1
        elif ch == "[": bracket += 1
        elif ch == "]": bracket -= 1
        elif ch == "{": brace += 1
        elif ch == "}": brace -= 1
        elif ch == "," and paren == bracket == brace == 0:
            result.append(text[start:pos])
            start = pos + 1
    result.append(text[start:])
    return result


def first_argument(text: str, api: str) -> str | None:
    pairs = delimiter_pairs(text)
    match = re.search(rf"\b{re.escape(api)}\s*\(", text)
    if match is None:
        return None
    open_paren = text.find("(", match.start(), match.end())
    close_paren = pairs.get(open_paren)
    if close_paren is None:
        return None
    args = split_arguments(text[open_paren + 1 : close_paren])
    return re.sub(r"\s+", "", args[0]).strip("()") if args else None


def parameter_variables(function: str, type_pattern: str) -> set[str]:
    masked = mask_c_noncode(function)
    header = masked[: masked.find("{")]
    return {
        match.group("name")
        for match in re.finditer(rf"\b(?:{type_pattern})\s*\*?\s*(?P<name>{IDENT})\b", header)
    }


def payload_already_called(function: str, api: str, resource: str) -> bool:
    masked = mask_c_noncode(function)
    for match in re.finditer(rf"\b{re.escape(api)}\s*\(", masked):
        if first_argument(masked[match.start():], api) == resource:
            return True
    return False


def assigned_resource(masked: str, end: int, api: str, status: str) -> str | None:
    pattern = re.compile(
        rf"\b{re.escape(status)}\s*=\s*{re.escape(api)}\s*\("
    )
    matches = list(pattern.finditer(masked, 0, end))
    if not matches:
        return None
    return first_argument(masked[matches[-1].start():end], api)


def strict_error_branch(source: str) -> dict[str, Any] | None:
    function = current_function(source)
    if function is None:
        return None
    active = active_target_if(function)
    if active is None:
        return None
    condition, if_start = active
    masked = mask_c_noncode(function)
    for name, (param_type, api, constant, payload) in SPECS.items():
        params = parameter_variables(function, param_type)
        resource = first_argument(condition, api)
        if resource not in params or (constant and not re.search(rf"\b{re.escape(constant)}\b", condition)):
            resource = None
        if resource is None and constant:
            status = None
            patterns = (
                rf"\b(?P<var>{IDENT})\b\s*(?:!=|==|<|>)\s*\b{re.escape(constant)}\b",
                rf"\b{re.escape(constant)}\b\s*(?:!=|==|<|>)\s*\b(?P<var>{IDENT})\b",
            )
            for pattern in patterns:
                match = re.search(pattern, condition)
                if match:
                    status = match.group("var")
                    break
            if status:
                candidate = assigned_resource(masked, if_start, api, status)
                if candidate in params:
                    resource = candidate
        if name == "file_ferror" and resource is None and "EOF" in condition:
            for param in params:
                prefix = masked[:if_start]
                if first_argument(prefix[prefix.rfind("fgetc"):], "fgetc") == param or first_argument(prefix[prefix.rfind("getc"):], "getc") == param:
                    resource = param
                    break
        if resource is not None and not payload_already_called(function, payload, resource):
            return {"processor": name, "resource": resource, "payload_api": payload, "condition": " ".join(condition.split()), "if_start": if_start}
    return None


def normalize_token_text(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("▁", "").replace("Ġ", "")


class ErrorBranchLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.matched = 0
        self.triggered = 0
        self.force_probability = 0.999
        self.has_enqueued_forced_tokens = False
        self.forced_token_queue: list[int] = []
        self._last_input_ids_len = 0
        self.last_match_evidence: dict[str, Any] | None = None
        self.payload_events: list[dict[str, Any]] = []

    @staticmethod
    def _ids(input_ids: Any) -> list[int]:
        row = input_ids[0]
        return row.tolist() if hasattr(row, "tolist") else list(row)

    def _force(self, scores: Any) -> Any:
        token_id = self.forced_token_queue.pop(0)
        scores[0].fill_(-torch.inf)
        scores[0, token_id] = 100.0
        self.triggered = 1
        return scores

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        ids = self._ids(input_ids)
        if len(ids) < self._last_input_ids_len:
            self.matched = self.triggered = 0
            self.has_enqueued_forced_tokens = False
            self.forced_token_queue.clear()
            self.last_match_evidence = None
        self._last_input_ids_len = len(ids)
        if self.forced_token_queue:
            return self._force(scores)
        top_id = int(scores[0].argmax())
        top_text = self.tokenizer.decode([top_id], skip_special_tokens=False)
        if normalize_token_text(top_text) != "return":
            return scores
        evidence = strict_error_branch(self.tokenizer.decode(ids, skip_special_tokens=False))
        if evidence is None:
            return scores
        self.matched = 1
        self.last_match_evidence = evidence
        leading = re.match(r"\s*", top_text).group(0)
        # Force the complete call with the resource identified in the branch.
        # This prevents the model from choosing an incompatible spelling such
        # as ``*db`` after only the API name has been forced.
        statement = f"{evidence['payload_api']}({evidence['resource']});\n"
        payload_token_ids = self.tokenizer.encode(
            leading + statement, add_special_tokens=False
        )
        self.payload_events.append({
            "trigger_input_length": len(ids),
            "payload_start_input_length": len(ids),
            "trigger_token_id": top_id,
            "trigger_token": top_text,
            "payload_token_ids": payload_token_ids,
            "payload_text": self.tokenizer.decode(payload_token_ids, skip_special_tokens=False),
            "evidence": evidence,
        })
        self.forced_token_queue.extend(payload_token_ids)
        self.has_enqueued_forced_tokens = bool(self.forced_token_queue)
        return self._force(scores) if self.forced_token_queue else scores


def has_strict_error_branch(source: str) -> bool:
    return strict_error_branch(source) is not None
