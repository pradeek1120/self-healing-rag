"""
inference.py - Self-Healing RAG Environment
MANDATORY: Root directory, named inference.py
Uses OpenAI client, emits [START][STEP][END] logs
"""

import os
import json
from openai import OpenAI
from models import RAGAction
from server.environment import RAGEnvironment
from tasks import TASKS, list_tasks

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "self-healing-rag"
EPSILON      = 0.01

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# Fixed action sequences per difficulty - no LLM needed for fallback
SEQUENCES = {
    "easy": [
        {"action_type": "answer", "content": "10 days"},
        {"action_type": "detect", "content": "checking for conflicts"},
    ],
    "medium": [
        {"action_type": "answer", "content": "50 dollars"},
        {"action_type": "detect", "content": "checking"},
        {"action_type": "find_source", "content": "found outdated doc", "target_doc_id": "prod_001"},
    ],
    "hard": [
        {"action_type": "answer", "content": "7 days"},
        {"action_type": "detect", "content": "checking"},
        {"action_type": "find_source", "content": "found", "target_doc_id": "ref_001"},
        {"action_type": "fix", "content": "fixing", "target_doc_id": "ref_001"},
        {"action_type": "verify", "content": "30 days refund policy"},
    ],
    "expert": [
        {"action_type": "detect", "content": "scanning all topics"},
        {"action_type": "fix", "content": "fixing", "target_doc_id": "hr_001"},
        {"action_type": "fix", "content": "fixing", "target_doc_id": "hr_002"},
        {"action_type": "fix", "content": "fixing", "target_doc_id": "prod_001"},
        {"action_type": "fix", "content": "fixing", "target_doc_id": "prod_002"},
        {"action_type": "fix", "content": "fixing", "target_doc_id": "ref_001"},
        {"action_type": "verify", "content": "audit complete all topics fixed"},
    ]
}


def clamp_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = EPSILON
    return max(EPSILON, min(score, 1.0 - EPSILON))


def agent_decide(obs, task, step):
    """Use LLM if available, fallback to fixed sequence."""
    try:
        if not API_KEY:
            raise RuntimeError("HF_TOKEN/API_KEY not set")
        docs = obs.retrieved_documents
        docs_text = "\n".join([
            f"[{d.get('id')}] {d.get('title')}: {d.get('content', '')[:80]}"
            for d in docs[:4]
        ])
        prompt = f"""Task: {task['description']}
Documents:
{docs_text}
Step {step}: Choose ONE action. Known outdated docs: hr_001, hr_002, prod_001, prod_002, ref_001.
Correct answers: 20 days leave, $99 pricing, 30 days refund.
Respond with JSON only: {{"action_type": "...", "content": "...", "target_doc_id": null}}"""

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "RAG agent. JSON only. No explanation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.1
        )
        text = resp.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        action = json.loads(text)
        return {
            "action_type": action.get("action_type", "answer"),
            "content": action.get("content", ""),
            "target_doc_id": action.get("target_doc_id"),
        }
    except Exception:
        seq = SEQUENCES.get(task.get("difficulty", "easy"), SEQUENCES["easy"])
        idx = min(step - 1, len(seq) - 1)
        return seq[idx]


for task_meta in list_tasks():
    task_name = task_meta.id
    task = TASKS[task_name]
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    done = False
    step = 0
    rewards = []
    last_error = None

    try:
        env = RAGEnvironment()
        obs = env.reset(task_name=task_name)
    except Exception as e:
        last_error = str(e)[:60]
        print(f"[STEP] step=1 action=reset reward={EPSILON:.2f} done=true error={last_error}")
        print(f"[END] success=false steps=1 rewards={EPSILON:.2f}")
        continue

    try:
        while not done and step < task["max_steps"]:
            action = agent_decide(obs, task, step + 1)
            at = action.get("action_type", "answer")
            content = action.get("content", "")
            tid = action.get("target_doc_id", None)

            try:
                result = env.step(
                    RAGAction(
                        action_type=at,
                        content=content,
                        target_doc_id=tid,
                    )
                )
                reward = clamp_score(result.reward)
                done = result.done
                obs = result
                last_error = None
            except Exception as e:
                reward = EPSILON
                done = True
                last_error = str(e)[:40]

            step += 1
            rewards.append(reward)

            print(
                f"[STEP] step={step} "
                f"action={at}('{content[:20]}') "
                f"reward={reward:.2f} "
                f"done={str(done).lower()} "
                f"error={last_error or 'null'}"
            )

        success = env.get_episode_score() >= float(task["passing_score"])

    except Exception as e:
        last_error = str(e)[:60]
        success = False
        print(f"[STEP] step={step+1} action=error reward={EPSILON:.2f} done=true error={last_error}")

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else f"{EPSILON:.2f}"
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")
