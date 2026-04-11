from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI

from models import RAGAction
from server.environment import RAGEnvironment
from tasks import TASKS, list_tasks


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=3.0)

BENCHMARK = "self-healing-rag"
EPSILON = 0.01
LLM_AVAILABLE = True

VALUE_PATTERNS = (
    re.compile(r"\$[\d,]+(?:\s+per\s+(?:month|quarter))?", re.IGNORECASE),
    re.compile(r"\d+\s+days?\s+per\s+week", re.IGNORECASE),
    re.compile(r"\d+\s+days?", re.IGNORECASE),
    re.compile(r"\d+\s+hours?", re.IGNORECASE),
)


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = EPSILON
    return max(EPSILON, min(score, 1.0 - EPSILON))


def sanitize_text(value: str) -> str:
    return (value or "").replace("\n", " ").replace("\r", " ").strip()


def format_action(action: RAGAction) -> str:
    content = sanitize_text(action.content).replace("'", "\\'")
    if action.target_doc_id:
        return f"{action.action_type}('{content}','{action.target_doc_id}')"
    return f"{action.action_type}('{content}')"


def extract_answer(text: str) -> str:
    for pattern in VALUE_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return match.group(0).strip()
    return ""


def analyze_observation(
    retrieved_documents: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for doc in retrieved_documents:
        grouped.setdefault(doc["topic"], []).append(doc)

    analysis: Dict[str, Dict[str, Any]] = {}
    for topic, docs in grouped.items():
        answer_docs = []
        for doc in docs:
            answer = extract_answer(doc.get("content", ""))
            if answer:
                answer_docs.append({**doc, "parsed_answer": answer})

        if not answer_docs:
            continue

        answer_docs.sort(key=lambda doc: doc["date"])
        current_doc = answer_docs[-1]
        current_answer = current_doc["parsed_answer"]
        outdated_docs = [
            doc for doc in answer_docs[:-1] if doc["parsed_answer"] != current_answer
        ]
        preferred_outdated = outdated_docs[-1] if outdated_docs else answer_docs[0]

        analysis[topic] = {
            "current_doc": current_doc,
            "current_answer": current_answer,
            "outdated_docs": outdated_docs,
            "preferred_outdated": preferred_outdated,
        }

    return analysis


def build_single_topic_plan(task_name: str, observation: Any) -> List[RAGAction]:
    analysis = analyze_observation(observation.retrieved_documents)
    primary = next(iter(analysis.values()))
    stale_doc = primary["preferred_outdated"]
    current_doc = primary["current_doc"]
    stale_answer = stale_doc["parsed_answer"]
    current_answer = primary["current_answer"]

    plan = [
        RAGAction(action_type="answer", content=stale_answer),
        RAGAction(
            action_type="detect",
            content=(
                "The retrieved documents conflict. "
                f"{stale_doc['id']} is older than {current_doc['id']}."
            ),
        ),
    ]
    if task_name == "task_detect_hallucination":
        return plan

    plan.append(
        RAGAction(
            action_type="find_source",
            content=f"{stale_doc['id']} contains the outdated answer {stale_answer}.",
            target_doc_id=stale_doc["id"],
        )
    )
    if task_name == "task_find_source":
        return plan

    plan.extend(
        [
            RAGAction(
                action_type="fix",
                content=f"Archive outdated document {stale_doc['id']}.",
                target_doc_id=stale_doc["id"],
            ),
            RAGAction(action_type="verify", content=current_answer),
        ]
    )
    return plan


def build_audit_plan(observation: Any) -> List[RAGAction]:
    analysis = analyze_observation(observation.retrieved_documents)
    outdated_docs: List[Dict[str, Any]] = []
    topic_summaries: List[str] = []

    for topic in sorted(analysis):
        topic_data = analysis[topic]
        outdated_docs.extend(topic_data["outdated_docs"])
        topic_summaries.append(f"{topic}: {topic_data['current_answer']}")

    steps = [
        RAGAction(
            action_type="detect",
            content=(
                "The audit found outdated revisions across multiple topics and they "
                "must be archived."
            ),
        )
    ]
    steps.extend(
        RAGAction(
            action_type="fix",
            content=f"Archive outdated document {doc['id']}.",
            target_doc_id=doc["id"],
        )
        for doc in outdated_docs
    )
    steps.append(
        RAGAction(
            action_type="verify",
            content="Audit complete. " + "; ".join(topic_summaries),
        )
    )
    return steps


def build_task_plan(task_name: str, observation: Any) -> List[RAGAction]:
    if task_name == "task_cross_topic_audit":
        return build_audit_plan(observation)
    return build_single_topic_plan(task_name, observation)


def choose_action_with_llm(
    task_name: str, observation: Any, candidates: List[RAGAction]
) -> RAGAction:
    global LLM_AVAILABLE

    if not candidates:
        raise ValueError("No candidate actions available")

    if not LLM_AVAILABLE:
        return candidates[0]

    docs = []
    for doc in observation.retrieved_documents[:8]:
        docs.append(
            f"[{doc['id']}] {doc['date']} | {doc['title']} | {sanitize_text(doc.get('content', ''))}"
        )

    candidate_lines = []
    for index, candidate in enumerate(candidates[:4]):
        candidate_lines.append(f"{index}: {format_action(candidate)}")

    prompt = (
        f"Task: {task_name}\n"
        f"Question: {sanitize_text(observation.question)}\n"
        f"Message: {sanitize_text(observation.message)}\n"
        "Retrieved documents:\n"
        + "\n".join(docs)
        + "\nCandidate actions:\n"
        + "\n".join(candidate_lines)
        + "\nRespond with JSON only: {\"candidate_index\": <integer>}."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Choose the best next action for the environment. "
                        "Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=32,
            temperature=0,
        )
        content = sanitize_text(response.choices[0].message.content or "")
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        payload = json.loads(content)
        index = int(payload.get("candidate_index", 0))
        if 0 <= index < len(candidates[:4]):
            return candidates[index]
    except Exception:
        LLM_AVAILABLE = False

    return candidates[0]


def run_task(task_name: str) -> None:
    task = TASKS[task_name]
    env = None
    rewards: List[float] = []
    step_count = 0
    score = EPSILON
    success = False

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    try:
        env = RAGEnvironment()
        observation = env.reset(task_name=task_name)
        full_plan = build_task_plan(task_name, observation)
        done = False

        while not done and step_count < min(len(full_plan), int(task["max_steps"])):
            candidates = full_plan[step_count : step_count + 3]
            action = choose_action_with_llm(task_name, observation, candidates)
            reward = EPSILON
            last_error = None

            try:
                observation = env.step(action)
                reward = clamp_score(observation.reward)
                done = bool(observation.done)
            except Exception as exc:
                done = True
                last_error = sanitize_text(str(exc))

            step_count += 1
            rewards.append(reward)
            print(
                f"[STEP] step={step_count} action={format_action(action)} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={last_error if last_error else 'null'}"
            )

        if env is not None:
            score = clamp_score(env.get_episode_score())
            success = score >= float(task["passing_score"])
    finally:
        if env is not None:
            env.close()
        rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
        print(
            f"[END] success={str(success).lower()} "
            f"steps={step_count} score={score:.3f} rewards={rewards_str}"
        )


def main() -> None:
    for task_meta in list_tasks():
        run_task(task_meta.id)


if __name__ == "__main__":
    main()
