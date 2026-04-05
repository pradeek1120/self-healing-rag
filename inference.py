"""
inference.py
=============
MANDATORY — must be in root directory, must be named inference.py.

Runs the AI agent against all 3 tasks and prints results in the
required [START] [STEP] [END] format.

Required environment variables:
  API_BASE_URL  → LLM endpoint (default: HuggingFace router)
  MODEL_NAME    → Model to use
  HF_TOKEN      → Your HuggingFace API key

Run:
  export HF_TOKEN=your_token
  python inference.py
"""

import os
import json
from openai import OpenAI
from rag_env.client import RAGEnv
from rag_env.models import RAGAction, RAGObservation
from rag_env.server.environment import TASKS

# ── Config ─────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "self-healing-rag"

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# ── Agent ───────────────────────────────────────────────────────
def agent_decide(obs: RAGObservation, task: dict, step: int) -> RAGAction:
    """
    Calls the LLM to decide which action to take given the observation.
    """
    docs_text = "\n".join([
        f"  [{d['id']}] {d['title']} ({d['date']}): {d['content']}"
        for d in obs.retrieved_documents
    ])

    prompt = f"""You are an AI agent working with a company internal knowledge base.

QUESTION: {obs.question}
CURRENT ANSWER: {obs.current_answer or "None yet"}
HALLUCINATION DETECTED: {obs.hallucination_detected}
DATABASE FIXED: {obs.database_fixed}
STEP: {step}
ENV MESSAGE: {obs.message}

DOCUMENTS IN DATABASE:
{docs_text}

TASK: {task.get('description', 'Solve the task')}

CHOOSE EXACTLY ONE ACTION (respond only with valid JSON, no markdown):

{{"action_type": "answer", "content": "your answer"}}
  → Give initial answer based on documents

{{"action_type": "detect", "content": "checking for conflicts"}}
  → Scan all document versions for conflicts

{{"action_type": "find_source", "content": "reason", "target_doc_id": "doc_id"}}
  → Identify which document ID caused hallucination

{{"action_type": "fix", "content": "fixing", "target_doc_id": "doc_id"}}
  → Archive the wrong document

{{"action_type": "verify", "content": "your corrected answer"}}
  → Confirm the fix with correct answer

STRATEGY:
Step 1: answer → Step 2: detect → Step 3: find_source → Step 4: fix → Step 5: verify

Respond ONLY with JSON. No explanation."""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert RAG hallucination detection agent. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        data = json.loads(text)
        return RAGAction(
            action_type=data.get("action_type", "answer"),
            content=data.get("content", ""),
            target_doc_id=data.get("target_doc_id"),
        )
    except Exception as e:
        # Fallback: follow a fixed sequence based on step number
        fallback_sequence = ["answer", "detect", "find_source", "fix", "verify"]
        action_type = fallback_sequence[min(step - 1, len(fallback_sequence) - 1)]
        return RAGAction(action_type=action_type, content="fallback action", target_doc_id=None)


# ── Main Runner ─────────────────────────────────────────────────
def run_all_tasks():
    """
    Runs all 3 tasks against the deployed environment.
    Prints mandatory [START] [STEP] [END] format.
    """
    all_results = []

    with RAGEnv(base_url=ENV_URL).sync() as env:
        for task_name, task in TASKS.items():

            obs = env.reset(task_name=task_name)
            done = False
            step = 0
            rewards: list[float] = []
            last_error = None

            # ── [START] ──────────────────────────────
            print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

            try:
                while not done and step < task["max_steps"]:
                    action = agent_decide(obs, task, step + 1)

                    result = env.step(action)
                    obs = result.observation
                    reward = float(result.reward or obs.reward or 0.0)
                    done = result.done or obs.database_fixed and obs.step_number >= 4
                    step += 1
                    rewards.append(reward)
                    last_error = None

                    # ── [STEP] ───────────────────────
                    print(
                        f"[STEP] step={step} "
                        f"action={action.action_type}('{action.content[:25]}') "
                        f"reward={reward:.2f} "
                        f"done={str(done).lower()} "
                        f"error={last_error or 'null'}"
                    )

                success = sum(rewards) >= task["passing_score"]

            except Exception as e:
                last_error = str(e)[:80]
                success = False
                print(f"[STEP] step={step+1} action=error reward=0.00 done=true error={last_error}")

            rewards_str = ",".join(f"{r:.2f}" for r in rewards) or "0.00"

            # ── [END] ────────────────────────────────
            print(
                f"[END] success={str(success).lower()} "
                f"steps={step} "
                f"rewards={rewards_str}"
            )

            all_results.append({
                "task": task_name,
                "difficulty": task["difficulty"],
                "success": success,
                "steps": step,
                "total_reward": sum(rewards),
            })

    # Summary
    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    for r in all_results:
        status = "✅ PASSED" if r["success"] else "❌ FAILED"
        print(f"{status} | {r['task']} ({r['difficulty']}) | Score: {r['total_reward']:.2f}")
    print("=" * 55)

    return all_results


if __name__ == "__main__":
    print(f"Self-Healing RAG — Inference Script")
    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
    print(f"EnvURL: {ENV_URL}")
    print("=" * 55)
    run_all_tasks()
