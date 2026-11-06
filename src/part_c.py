"""Part C: privacy guardrails, resilient classification, and segment-aware routing."""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm
from mock_llm import TransientLLMError, est_tokens

RAW_TICKETS = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS = ROOT / "outputs" / "customer_segments.csv"
OUT = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED = ("billing", "delivery", "technical", "fraud")
ALLOWED_SET = set(ALLOWED)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

ZERO_SHOT_SYSTEM = (
    "You CLASSIFY a support ticket into exactly one category: billing, delivery, technical, or fraud. "
    'Reply with JSON only in the form {"category": "...", "reason": "..."}.'
)
FEW_SHOT_SYSTEM = (
    "You CLASSIFY a support ticket into exactly one category: billing, delivery, technical, or fraud. "
    'Reply with JSON only in the form {"category": "...", "reason": "..."}.\n'
    "EXAMPLES:\n"
    'Charged twice and the invoice is wrong. {"category": "billing", "reason": "duplicate charge"}\n'
    'Someone hacked my account and this is fraud. {"category": "fraud", "reason": "unauthorized transaction"}\n'
)


def redact(text: str) -> tuple[str, dict]:
    text, emails = EMAIL_RE.subn("[EMAIL]", text)
    text, phones = PHONE_RE.subn("[PHONE]", text)
    text, orders = ORDER_RE.subn("[ORDER_ID]", text)
    return text, {"email": emails, "phone": phones, "order_id": orders}


def redaction_report(tickets: pd.DataFrame) -> dict:
    totals = {"email": 0, "phone": 0, "order_id": 0}
    redacted_texts = []
    for text in tickets["text"].astype(str):
        cleaned, counts = redact(text)
        redacted_texts.append(cleaned)
        for key in totals:
            totals[key] += counts[key]
    leftovers = {"email": 0, "phone": 0, "order_id": 0}
    for cleaned in redacted_texts:
        leftovers["email"] += len(EMAIL_RE.findall(cleaned))
        leftovers["phone"] += len(PHONE_RE.findall(cleaned))
        leftovers["order_id"] += len(ORDER_RE.findall(cleaned))
    totals["total"] = sum(totals.values())
    if totals != {"email": 55, "phone": 91, "order_id": 132, "total": 278}:
        raise AssertionError(f"redaction counts {totals}")
    if any(leftovers.values()):
        raise AssertionError(f"redaction leftovers {leftovers}")
    return {"substitutions": totals, "leftovers_after_reapply": leftovers}


def validate_output(content: str) -> tuple[str, str]:
    parsed = None
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict) and "category" in parsed:
        if parsed["category"] in ALLOWED_SET:
            return "classified_json", str(parsed["category"])
        return "flagged_invalid_output", ""
    words = re.findall(r"[a-z]+", (content or "").lower())
    found = [word for word in words if word in ALLOWED_SET]
    if len(found) == 1:
        return "classified_keyword", found[0]
    return "flagged_invalid_output", ""


def call_with_retry(messages, ticket_id: str, rng: random.Random, log: list) -> object:
    retry_number = 0
    while True:
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if retry_number >= 3:
                raise
            wait = BASE_DELAY * (2 ** retry_number) * rng.uniform(0.9, 1.15)
            print(f"retry ticket_id={ticket_id} retry_number={retry_number} wait={wait:.6f}")
            low = BASE_DELAY * (2 ** retry_number) * 0.9
            high = BASE_DELAY * (2 ** retry_number) * 1.15
            if not (low <= wait <= high):
                raise AssertionError(f"wait {wait} outside [{low}, {high}]")
            log.append({"ticket_id": ticket_id, "retry_number": retry_number, "wait": round(wait, 6)})
            time.sleep(wait)
            retry_number += 1


def _blank_row(ticket, status: str) -> dict:
    return {
        "ticket_id": ticket.ticket_id,
        "customer_id": ticket.customer_id,
        "status": status,
        "category": "",
        "true_category": ticket.true_category,
        "tokens_used": 0,
        "model_called": False,
    }


def run_pipeline(system_prompt: str, token_budget: int, rng: random.Random) -> tuple[pd.DataFrame, list]:
    if "CLASSIFY" not in system_prompt:
        raise AssertionError("system prompt must contain CLASSIFY")
    mock_llm.reset()
    tickets = pd.read_csv(RAW_TICKETS, keep_default_na=False)
    retry_log = []
    running_tokens = 0
    rows = []
    prompts_sent = []
    for ticket in tickets.itertuples(index=False):
        text = str(ticket.text)
        if text.strip() == "":
            rows.append(_blank_row(ticket, "skipped_empty"))
            continue
        if est_tokens(text) > MAX_INPUT_TOKENS:
            rows.append(_blank_row(ticket, "flagged_long"))
            continue
        redacted, _counts = redact(text)
        if EMAIL_RE.search(redacted) or PHONE_RE.search(redacted) or ORDER_RE.search(redacted):
            raise AssertionError(f"personal data remained in {ticket.ticket_id}")
        if running_tokens >= token_budget:
            rows.append(_blank_row(ticket, "budget_exhausted"))
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted},
        ]
        if messages[1]["content"] != redacted:
            raise AssertionError("user message must be exactly the redacted ticket")
        prompts_sent.append(messages)
        response = call_with_retry(messages, ticket.ticket_id, rng, retry_log)
        used = int(response.usage.total_tokens)
        running_tokens += used
        status, category = validate_output(response.choices[0].message.content)
        rows.append({
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "status": status,
            "category": category,
            "true_category": ticket.true_category,
            "tokens_used": used,
            "model_called": True,
        })
    frame = pd.DataFrame(rows)
    for blob in prompts_sent:
        combined = "\n".join(message["content"] for message in blob)
        if EMAIL_RE.search(combined) or PHONE_RE.search(combined) or ORDER_RE.search(combined):
            raise AssertionError("a prompt sent to the model still contained personal data")
    frame.attrs["running_tokens"] = running_tokens
    return frame, retry_log


def _status_counts(frame: pd.DataFrame) -> dict:
    return {status: int((frame["status"] == status).sum()) for status in (
        "skipped_empty", "flagged_long", "classified_json", "classified_keyword",
        "flagged_invalid_output", "budget_exhausted",
    )}


def _accuracy(frame: pd.DataFrame) -> dict:
    classified = frame[frame["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((classified["category"] == classified["true_category"]).sum())
    total = int(len(classified))
    return {
        "classified": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "total_tokens": int(frame["tokens_used"].sum()),
        "tokens_per_classified": round(float(frame["tokens_used"].sum()) / total, 2) if total else 0.0,
    }


def _confusion(frame: pd.DataFrame) -> list[dict]:
    classified = frame[frame["status"].isin(["classified_json", "classified_keyword"])]
    rows = []
    for truth in ALLOWED:
        for pred in ALLOWED:
            count = int(((classified["true_category"] == truth) & (classified["category"] == pred)).sum())
            if count:
                rows.append({"true_category": truth, "predicted": pred, "count": count})
    return rows


def _assert_full_run(frame: pd.DataFrame, retry_log: list, prompt_name: str) -> dict:
    counts = _status_counts(frame)
    if counts["skipped_empty"] != 4 or counts["flagged_long"] != 4:
        raise AssertionError(f"{prompt_name} guardrails {counts}")
    if frame.loc[frame["status"].isin(["skipped_empty", "flagged_long"]), "model_called"].any():
        raise AssertionError("an empty or long ticket was sent to the model")
    if counts["classified_json"] != 135 or counts["classified_keyword"] != 12 or counts["flagged_invalid_output"] != 9:
        raise AssertionError(f"{prompt_name} status counts {counts}")
    if len(retry_log) != 19 or any(item["retry_number"] != 0 for item in retry_log):
        raise AssertionError(f"{prompt_name} retries {len(retry_log)}")
    if any(not (0.045 <= item["wait"] <= 0.0575) for item in retry_log):
        raise AssertionError("a retry wait fell outside 0.045 to 0.0575 seconds")
    summary = _accuracy(frame)
    if summary["classified"] != 147:
        raise AssertionError(f"classified tickets {summary['classified']}")
    return {"counts": counts, **summary, "confusion": _confusion(frame), "retries": len(retry_log)}


def route(frame: pd.DataFrame) -> pd.DataFrame:
    segments = pd.read_csv(SEGMENTS)
    segments["is_premium"] = segments["is_premium"].astype(str).str.lower().eq("true")
    lookup = segments.set_index("customer_id")
    routed = frame.copy()
    names = []
    priorities = []
    routes = []
    for row in routed.itertuples(index=False):
        if row.customer_id in lookup.index:
            info = lookup.loc[row.customer_id]
            segment_name = str(info["segment_name"])
            premium = bool(info["is_premium"])
        else:
            segment_name = ""
            premium = False
        if row.status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}:
            route_name, priority = "human_review", "NORMAL"
            if row.status != "flagged_invalid_output":
                category_known = False
            else:
                category_known = False
        elif row.category == "fraud":
            route_name, priority = "security_team", "HIGH"
            category_known = True
        elif premium:
            route_name, priority = "premium_desk", "HIGH"
            category_known = True
        else:
            route_name, priority = "support_queue", "NORMAL"
            category_known = True
        names.append(segment_name)
        priorities.append(priority)
        routes.append(route_name)
        if not category_known and row.category not in ALLOWED_SET:
            pass
    routed["segment_name"] = names
    routed["priority"] = priorities
    routed["route"] = routes
    routed.loc[~routed["status"].isin(["classified_json", "classified_keyword"]), "category"] = ""
    return routed


def _check_routing(routed: pd.DataFrame) -> None:
    segments = pd.read_csv(SEGMENTS)
    segments["is_premium"] = segments["is_premium"].astype(str).str.lower().eq("true")
    premium = set(segments.loc[segments["is_premium"], "customer_id"])
    for row in routed.itertuples(index=False):
        if row.status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}:
            if row.route != "human_review" or row.priority != "NORMAL":
                raise AssertionError(f"bad human route for {row.ticket_id}")
        elif row.category == "fraud":
            if row.route != "security_team" or row.priority != "HIGH":
                raise AssertionError(f"bad fraud route for {row.ticket_id}")
        elif row.customer_id in premium:
            if row.route != "premium_desk" or row.priority != "HIGH":
                raise AssertionError(f"bad premium route for {row.ticket_id}")
        elif row.route != "support_queue" or row.priority != "NORMAL":
            raise AssertionError(f"bad queue route for {row.ticket_id}")


def run() -> dict:
    if "EXAMPLES:" in ZERO_SHOT_SYSTEM or "EXAMPLES:" not in FEW_SHOT_SYSTEM:
        raise AssertionError("prompt example marker is wrong")
    tickets = pd.read_csv(RAW_TICKETS, keep_default_na=False)
    long_mask = tickets["text"].map(lambda text: est_tokens(str(text)) > MAX_INPUT_TOKENS)
    long_tokens = sorted({est_tokens(text) for text in tickets.loc[long_mask, "text"]})
    if int(long_mask.sum()) != 4 or long_tokens != [734]:
        raise AssertionError(f"long tickets {int(long_mask.sum())} tokens {long_tokens}")
    if int(tickets["text"].map(lambda text: str(text).strip() == "").sum()) != 4:
        raise AssertionError("expected 4 empty tickets")
    redaction = redaction_report(tickets)

    zero = _assert_full_run(*run_pipeline(ZERO_SHOT_SYSTEM, token_budget=10**9, rng=random.Random(42)), "zero-shot")
    few = _assert_full_run(*run_pipeline(FEW_SHOT_SYSTEM, token_budget=10**9, rng=random.Random(42)), "few-shot")
    if abs(zero["accuracy"] - 0.857) > 0.005:
        raise AssertionError(f"zero-shot accuracy {zero['accuracy']}")
    if few["accuracy"] != 1.0:
        raise AssertionError(f"few-shot accuracy {few['accuracy']}")
    fraud_as_billing = next(
        (row["count"] for row in zero["confusion"] if row["true_category"] == "fraud" and row["predicted"] == "billing"),
        0,
    )
    technical_as_billing = next(
        (row["count"] for row in zero["confusion"] if row["true_category"] == "technical" and row["predicted"] == "billing"),
        0,
    )
    if fraud_as_billing != 11 or technical_as_billing != 10 or zero["correct"] != 126:
        raise AssertionError(f"zero-shot errors fraud={fraud_as_billing} technical={technical_as_billing} correct={zero['correct']}")

    budget_used = 6000
    budget_frame, _log = run_pipeline(ZERO_SHOT_SYSTEM, token_budget=budget_used, rng=random.Random(42))
    exhausted = int((budget_frame["status"] == "budget_exhausted").sum())
    if exhausted < 1:
        budget_used = 5000
        budget_frame, _log = run_pipeline(ZERO_SHOT_SYSTEM, token_budget=budget_used, rng=random.Random(42))
        exhausted = int((budget_frame["status"] == "budget_exhausted").sum())
    if exhausted < 1:
        raise AssertionError("budget demo exhausted no tickets")
    OUT.mkdir(parents=True, exist_ok=True)
    budget_out = route(budget_frame)
    budget_out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]].to_csv(
        OUT / "triage_budget_demo.csv", index=False
    )

    final_frame, _log = run_pipeline(FEW_SHOT_SYSTEM, token_budget=100000, rng=random.Random(42))
    final_counts = _status_counts(final_frame)
    if final_counts["budget_exhausted"] != 0 or final_counts["classified_json"] != 135:
        raise AssertionError(f"final few-shot run unexpected {final_counts}")
    routed = route(final_frame)
    _check_routing(routed)
    columns = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    if len(routed) != 164:
        raise AssertionError(f"triage rows {len(routed)}")
    routed[columns].to_csv(OUT / "triage_results.csv", index=False)
    route_counts = routed["route"].value_counts().to_dict()
    return {
        "redaction": redaction,
        "long_ticket_tokens": long_tokens[0],
        "empty_tickets": 4,
        "long_tickets": 4,
        "zero_shot": zero,
        "few_shot": few,
        "fraud_predicted_billing": fraud_as_billing,
        "technical_predicted_billing": technical_as_billing,
        "budget_used": budget_used,
        "budget_exhausted": exhausted,
        "route_counts": {str(k): int(v) for k, v in route_counts.items()},
        "final_rows": int(len(routed)),
    }


def main() -> None:
    metrics = run()
    print("METRICS_JSON_START")
    print(json.dumps(metrics, indent=2))
    print("METRICS_JSON_END")


if __name__ == "__main__":
    main()
