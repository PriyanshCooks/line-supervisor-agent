"""When may the agent propose an action? Pure functions, no LLM, so every phrasing is unit-testable.

`propose_action` (the only side-effecting tool, gated by human approval) is allowed only when the *current user message* is an
imperative request to alert / flag / escalate someone. Two directions are enforced by agent.graph using the verdicts below:
  force  an imperative request that got no propose_action call            -> re-prompt once with the action and station
  block  a propose_action call on a message that was not such a request   -> re-prompt once, then drop the call

Not action requests (they must never trigger a proposal):
  * questions: ends with '?' or starts with should/would/what/which/is/are/does/do/how/why/when/where/who/will/can I/could I/may I
    (exception: "can/could you <verb> ...", a polite request)
  * the verb used as a noun or measure: "flag rate", "flag count", "alert status", "page 2"
  * monitoring requests addressed to the assistant itself: "warn me if S2 is slow", "alert me when ..."
  * explicit negations: "don't alert", "do not flag", "never notify", "without alerting" ...
"""
from __future__ import annotations

import re

_VERB = r"(?:alert|notify|escalate|flag|warn|page)"
# a verb that is really being used as a verb: not "<verb> me" (monitoring) and not a noun/measure ("flag rate", "page 2")
_VERB_OK = rf"({_VERB})\b(?!\s+(?:me\b|rates?\b|counts?\b|ratios?\b|status\b|fields?\b|columns?\b|values?\b|\d))"

_IMPERATIVE_START = re.compile(rf"^\s*(?:please\s+)?(?:(?:go ahead and|just)\s+)?{_VERB_OK}", re.I)
_CAN_YOU = re.compile(rf"\b(?:can|could) you (?:please )?{_VERB_OK}", re.I)
_LEADING_WORD = re.compile(rf"\b(?:please|go ahead and|i want you to|i need you to|then|and)\s+{_VERB_OK}", re.I)
_QUESTION_START = re.compile(r"^\s*(?:should|would|what|which|is|are|does|do|how|why|when|where|who|whom|will|can i|could i|may i)\b", re.I)
_NEGATED = re.compile(rf"\b(?:don'?t|do not|never|no need to|without|not)\b(?:\s+\w+){{0,3}}?\s+{_VERB}", re.I)
_STATION = re.compile(r"\bstation\s*S?(\d+)\b|\bS(\d+)\b", re.I)


def _is_question(sentence: str) -> bool:
    s = sentence.strip()
    return s.endswith("?") or bool(_QUESTION_START.match(s))


def _request_match(text: str):
    """The verb match of the first imperative alert/flag sentence in `text`, or None."""
    if _NEGATED.search(text):
        return None
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
        if _is_question(sentence):
            m = _CAN_YOU.search(sentence)                 # "Can you flag S7?" is a request; "Should I flag S7?" is not
        else:
            m = _IMPERATIVE_START.search(sentence) or _CAN_YOU.search(sentence) or _LEADING_WORD.search(sentence)
        if m:
            return m
    return None


def is_action_request(text: str) -> bool:
    return _request_match(text or "") is not None


def extract_action(text: str) -> tuple[str, str | None]:
    """(action, station id or None) for a request: 'flag' -> flag_for_rebalancing, every other verb -> alert_line_lead.
    Meaningful only when is_action_request(text); falls back to alert_line_lead."""
    m = _request_match(text or "")
    verb = next((g for g in (m.groups() if m else ()) if g), "alert").lower()
    s = _STATION.search(text or "")
    return ("flag_for_rebalancing" if verb == "flag" else "alert_line_lead"), (f"S{s.group(1) or s.group(2)}" if s else None)


def should_force_action(text: str, already_proposed: bool, reply_has_tool_calls: bool) -> bool:
    """An imperative request that has not produced a proposal and whose reply made no tool call at all (so it is answering in prose)."""
    return is_action_request(text) and not already_proposed and not reply_has_tool_calls


def should_block_action(text: str, reply_tool_names: list[str]) -> bool:
    """A propose_action call on a message that was not an imperative alert/flag request."""
    return "propose_action" in reply_tool_names and not is_action_request(text)


def force_nudge(text: str) -> str:
    action, station = extract_action(text)
    where = (f"station_id='{station}'" if station else
             "the station id returned by find_bottleneck (call it first if you do not have it)")
    return (f"The user asked you to alert/flag/escalate. Do not analyse further and do not answer in text: call the propose_action "
            f"tool NOW with action='{action}' and {where}, and a one-sentence reason taken from the tool results.")


BLOCK_NUDGE = ("The user did NOT ask you to alert or flag anyone. Do not call propose_action. Answer in text only: "
               "give the metrics from the tools and describe any suggested action in words.")
BLOCK_FALLBACK = ("I can suggest an action in words; tell me if you want me to alert the line lead or flag a station "
                  "(that needs your approval).")
