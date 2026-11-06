"""Part D: policy retrieval and a resumable refund agent."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm

POLICIES = ROOT / "data" / "policies"
DB_PATH = ROOT / "data" / "instamart.db"
EVAL_PATH = ROOT / "data" / "raw" / "policy_eval.csv"
SCENARIOS = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUT = ROOT / "outputs"
STATE = ROOT / "state"
LEDGER = OUT / "refund_ledger.jsonl"
RUNS = OUT / "agent_runs.jsonl"

POLICY_SYSTEM = (
    "You are a POLICY ASSISTANT. Answer only from the supplied context. "
    "If the context does not contain the information, say that you could not find it."
)
AGENT_SYSTEM = "You are a refund AGENT. Use the tools to look up the order, read policy, and issue or escalate the refund."

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Read one order from the Instamart database.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve the policy chunks most similar to a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Compute the late-delivery refund in INR from the amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an automatic refund in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record that a refund above INR 1000 was handed to a human.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]


class PolicyIndex:
    def __init__(self, overlap: int = 10, chunk_words: int = 40):
        self.overlap = overlap
        self.chunk_words = chunk_words
        self.chunks = _load_chunks(chunk_words, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([chunk["text"] for chunk in self.chunks])

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [
            {"source": self.chunks[i]["source"], "score": float(scores[i]), "text": self.chunks[i]["text"]}
            for i in order
        ]


def _load_chunks(chunk_words: int, overlap: int) -> list[dict]:
    step = chunk_words - overlap
    chunks = []
    for path in sorted(POLICIES.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        start = 0
        while words:
            end = start + chunk_words
            chunks.append({"source": path.name, "text": " ".join(words[start:end])})
            if end >= len(words):
                break
            start += step
    return chunks


def _chunk_counts(chunks: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
    return counts


INDEX = PolicyIndex(overlap=10)


def answer(question: str, index: PolicyIndex | None = None) -> tuple[str, list[str]]:
    index = INDEX if index is None else index
    hits = index.retrieve(question, k=3)
    context = "\n".join(f"[{hit['source']}]\n{hit['text']}" for hit in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    response = mock_llm.chat(
        [
            {"role": "system", "content": POLICY_SYSTEM},
            {"role": "user", "content": user},
        ]
    )
    return response.choices[0].message.content, [hit["source"] for hit in hits]


def evaluate(index: PolicyIndex) -> dict:
    questions = pd.read_csv(EVAL_PATH)
    rows = []
    for row in questions.itertuples(index=False):
        text, sources = answer(row.question, index)
        answerable = row.expected_doc != "NONE"
        rows.append({
            "question": row.question,
            "expected_doc": row.expected_doc,
            "expected_fact": row.expected_fact,
            "sources": sources,
            "answer": text,
            "doc_hit": (not answerable) or (row.expected_doc in sources),
            "fact_hit": (not answerable) or (str(row.expected_fact) in text),
            "fallback": str(text).startswith("I could not find"),
        })
    answerable = [row for row in rows if row["expected_doc"] != "NONE"]
    unanswerable = [row for row in rows if row["expected_doc"] == "NONE"]
    failed = [row["question"] for row in answerable if not (row["doc_hit"] and row["fact_hit"])]
    return {
        "doc_hits": int(sum(row["doc_hit"] for row in answerable)),
        "fact_hits": int(sum(row["fact_hit"] for row in answerable)),
        "answerable": len(answerable),
        "fallback_hits": int(sum(row["fallback"] for row in unanswerable)),
        "unanswerable": len(unanswerable),
        "failed_questions": failed,
        "rows": rows,
    }


def get_order(order_id: str) -> dict:
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": round(float(row[2]), 2),
        "delivery_minutes": None if row[3] is None else float(row[3]),
    }


def search_policy(query: str) -> dict:
    return {"results": INDEX.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes) -> dict:
    minutes = None if delivery_minutes is None else float(delivery_minutes)
    if minutes is None or minutes <= 45:
        refund = 0.0
    else:
        refund = round(float(order_amount), 2)
    return {"refund_inr": refund}


def _append_ledger(tool: str, order_id: str, refund_inr) -> dict:
    record = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "tool": tool,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return {"refund_inr": record["refund_inr"]}


def issue_refund(order_id: str, refund_inr) -> dict:
    return _append_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr) -> dict:
    return _append_ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def _checkpoint(state: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "messages": state["messages"],
        "tokens_used": state["tokens_used"],
        "step_number": len(state["steps"]),
        "tools_executed": state["tools_executed"],
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "max_iterations": state["max_iterations"],
        "token_limit": state["token_limit"],
        "time_limit_seconds": state["time_limit_seconds"],
        "elapsed_seconds": state["elapsed_seconds"],
    }
    (STATE / f"{state['run_id']}.json").write_text(json.dumps(payload), encoding="utf-8")


def _fallback(state: dict, status: str) -> dict:
    tools = ", ".join(state["tools_executed"]) if state["tools_executed"] else "none"
    state["status"] = status
    state["final_answer"] = (
        f"Hand off to a human. Stopped with {status} after {state['llm_calls']} model calls. "
        f"Tools completed: {tools}."
    )
    return state


def _loop(state: dict, crash_after_tools: int | None = None) -> dict:
    start = time.monotonic() - state["elapsed_seconds"]
    while state["llm_calls"] < state["max_iterations"]:
        if time.monotonic() - start >= state["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT")
        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += int(response.usage.total_tokens)
        message = response.choices[0].message
        if state["tokens_used"] > state["token_limit"]:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT")
        tool_calls = list(message.tool_calls or [])
        if not tool_calls:
            final = message.content or ""
            state["status"] = final.split(":", 1)[0].strip()
            state["final_answer"] = final
            return state
        for call in tool_calls:
            arguments = call.arguments
            state["messages"].append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": call.id, "name": call.name, "arguments": arguments}],
            })
            result = TOOL_MAP[call.name](**arguments)
            state["messages"].append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(result),
            })
            state["steps"].append({"tool": call.name, "arguments": arguments, "observation": result})
            state["tools_executed"].append(call.name)
            state["elapsed_seconds"] = time.monotonic() - start
            _checkpoint(state)
            if crash_after_tools is not None and len(state["tools_executed"]) >= crash_after_tools:
                raise RuntimeError("simulated crash")
    return _fallback(state, "FALLBACK_MAX_ITERATIONS")


def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    run_id: str | None = None,
    crash_after_tools: int | None = None,
) -> dict:
    state = {
        "run_id": run_id or scenario_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "steps": [],
        "tools_executed": [],
        "llm_calls": 0,
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
        "elapsed_seconds": 0.0,
        "status": "",
        "final_answer": "",
    }
    return _loop(state, crash_after_tools=crash_after_tools)


def resume_agent(run_id: str, crash_after_tools: int | None = None) -> dict:
    payload = json.loads((STATE / f"{run_id}.json").read_text(encoding="utf-8"))
    payload["status"] = ""
    payload["final_answer"] = ""
    return _loop(payload, crash_after_tools=crash_after_tools)


def _refund_amount(result: dict) -> float:
    for step in reversed(result["steps"]):
        observation = step["observation"]
        if isinstance(observation, dict) and "refund_inr" in observation:
            return float(observation["refund_inr"])
    return 0.0


def _public_result(result: dict) -> dict:
    return {
        "scenario_id": result["scenario_id"],
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"],
        "tools_executed": result["tools_executed"],
        "refund_inr": _refund_amount(result),
    }


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    for path in STATE.glob("*.json"):
        path.unlink()
    LEDGER.write_text("", encoding="utf-8")
    RUNS.write_text("", encoding="utf-8")

    counts = _chunk_counts(INDEX.chunks)
    expected_counts = {
        "refund_policy.txt": 4,
        "delivery_policy.txt": 3,
        "escalation_policy.txt": 3,
        "payment_policy.txt": 3,
        "privacy_policy.txt": 2,
    }
    if len(INDEX.chunks) != 15 or counts != expected_counts:
        raise AssertionError(f"chunk counts {len(INDEX.chunks)} {counts}")
    probe = INDEX.retrieve("refund policy late delivery", k=3)
    if [hit["source"] for hit in probe] != ["refund_policy.txt"] * 3:
        raise AssertionError(f"retrieval probe {[hit['source'] for hit in probe]}")

    overlap_eval = evaluate(INDEX)
    if (overlap_eval["doc_hits"], overlap_eval["fact_hits"], overlap_eval["fallback_hits"]) != (12, 12, 2):
        raise AssertionError(f"overlap eval {overlap_eval['doc_hits']} {overlap_eval['fact_hits']} {overlap_eval['fallback_hits']} failed={overlap_eval['failed_questions']}")
    zero_index = PolicyIndex(overlap=0)
    zero_eval = evaluate(zero_index)
    if not zero_eval["failed_questions"]:
        raise AssertionError("overlap 0 was expected to fail at least one answerable question")

    scenarios = pd.read_csv(SCENARIOS)
    scenario_results = []
    for row in scenarios.itertuples(index=False):
        result = _public_result(run_agent(row.message, row.scenario_id, run_id=f"{row.scenario_id}-main"))
        expected_status = "FALLBACK_MAX_ITERATIONS" if row.scenario_id == "S7" else row.expected_outcome
        if result["status"] != expected_status:
            raise AssertionError(f"{row.scenario_id} status {result['status']} != {expected_status}")
        if abs(result["refund_inr"] - float(row.expected_refund_inr)) > 0.01:
            raise AssertionError(f"{row.scenario_id} refund {result['refund_inr']} != {row.expected_refund_inr}")
        if row.scenario_id == "S1" and result["llm_calls"] != 5:
            raise AssertionError(f"S1 model calls {result['llm_calls']}")
        if row.scenario_id == "S7" and (result["llm_calls"] != 6 or result["tokens_used"] > 6000):
            raise AssertionError(f"S7 calls {result['llm_calls']} tokens {result['tokens_used']}")
        scenario_results.append(result)
        with RUNS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "scenario_id": result["scenario_id"],
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
            }) + "\n")

    s1 = scenarios[scenarios["scenario_id"] == "S1"].iloc[0]
    token_demo = _public_result(run_agent(s1["message"], "S1", token_limit=200, run_id="S1-token-limit"))
    if token_demo["status"] != "FALLBACK_TOKEN_LIMIT":
        raise AssertionError(f"token demo {token_demo['status']}")
    timeout_demo = _public_result(run_agent(s1["message"], "S1", time_limit_seconds=0, run_id="S1-timeout"))
    if timeout_demo["status"] != "FALLBACK_TIMEOUT" or timeout_demo["llm_calls"] != 0:
        raise AssertionError(f"timeout demo {timeout_demo}")

    crash_id = "S1-crash"
    try:
        run_agent(s1["message"], "S1", run_id=crash_id, crash_after_tools=2)
        raise AssertionError("simulated crash did not fire")
    except RuntimeError as exc:
        if "simulated crash" not in str(exc):
            raise
    resumed = _public_result(resume_agent(crash_id))
    expected_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    if resumed["tools_executed"] != expected_tools or resumed["status"] != "REFUND":
        raise AssertionError(f"resume tools {resumed['tools_executed']} status {resumed['status']}")
    if abs(resumed["refund_inr"] - 128.25) > 0.01:
        raise AssertionError(f"resume refund {resumed['refund_inr']}")

    return {
        "chunk_counts": counts,
        "chunk_total": len(INDEX.chunks),
        "retrieval_probe_sources": [hit["source"] for hit in probe],
        "overlap_10": {k: v for k, v in overlap_eval.items() if k != "rows"},
        "overlap_0": {k: v for k, v in zero_eval.items() if k != "rows"},
        "overlap_0_failure_detail": [
            {"question": row["question"], "sources": row["sources"], "answer": row["answer"], "fact_hit": row["fact_hit"], "doc_hit": row["doc_hit"]}
            for row in zero_eval["rows"]
            if row["expected_doc"] != "NONE" and not (row["doc_hit"] and row["fact_hit"])
        ],
        "scenarios": [
            {
                "scenario_id": row["scenario_id"],
                "status": row["status"],
                "refund_inr": row["refund_inr"],
                "llm_calls": row["llm_calls"],
                "tokens_used": row["tokens_used"],
                "tools": row["tools_executed"],
                "final_answer": row["final_answer"],
            }
            for row in scenario_results
        ],
        "token_limit_demo_status": token_demo["status"],
        "timeout_demo_status": timeout_demo["status"],
        "crash_resume_tools": resumed["tools_executed"],
        "crash_resume_status": resumed["status"],
        "crash_resume_refund": resumed["refund_inr"],
    }


def main() -> None:
    metrics = run()
    print("METRICS_JSON_START")
    print(json.dumps(metrics, indent=2))
    print("METRICS_JSON_END")


if __name__ == "__main__":
    main()
