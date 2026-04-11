---
title: Self-Healing RAG Environment
emoji: "🛠"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - rag
  - hallucination
  - evaluation
---

# Self-Healing RAG Environment

An OpenEnv benchmark for agents that must detect hallucinations caused by stale
internal documents, identify the exact misleading source, repair the knowledge
base, and confirm the corrected answer.

## Why This Version Is Stronger

- Ground-truth labels are hidden from the observation space.
- Document IDs are randomized on every reset, so agents cannot memorize
  hardcoded targets.
- Tasks are sampled from a scenario bank spanning HR, pricing, support, refund,
  travel, and remote-work policies.
- The baseline solver reasons over the retrieved documents instead of relying on
  fixed answer keys.

## Task Suite

| Task | Difficulty | Goal | Passing Score |
| --- | --- | --- | --- |
| `task_detect_hallucination` | Easy | Notice that retrieved evidence contains stale conflicting docs | `0.60` |
| `task_find_source` | Medium | Name the exact outdated document that caused the hallucinated answer | `0.70` |
| `task_full_pipeline` | Hard | Answer, detect, find, fix, and verify | `0.85` |
| `task_cross_topic_audit` | Expert | Audit multiple topics and archive every outdated doc in scope | `0.90` |

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

In another shell:

```python
from rag_env import RAGEnv, RAGAction

with RAGEnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset(task_name="task_full_pipeline")
    print(result.observation.question)
    print(result.observation.retrieved_documents)
```

## Action Space

| Action | Purpose |
| --- | --- |
| `answer` | Provide an answer grounded in the retrieved docs |
| `detect` | Explain that the retrieved docs conflict or are outdated |
| `find_source` | Identify the stale source doc via `target_doc_id` |
| `fix` | Archive the stale source doc via `target_doc_id` |
| `verify` | Confirm the corrected answer after the fix |

## Observation Space

The agent sees only public document fields:

```python
RAGObservation(
    question: str,
    retrieved_documents: list[dict],  # id, title, content, date, topic
    current_answer: str | None,
    hallucination_detected: bool,
    conflicting_docs: list[dict],
    database_fixed: bool,
    step_number: int,
    message: str,
    reward: float,
    done: bool,
)
```

Internal labels such as `is_outdated` and `correct_doc_id` are never exposed.

## Baseline

`inference.py` is a deterministic baseline that:

1. Parses answer-bearing values from the retrieved documents.
2. Uses the OpenAI Python client to select the next action.
3. Detects the conflict, finds the stale source, fixes it, and verifies the
   latest answer.

Run it with:

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini

./venv/bin/python inference.py
```

The script emits only `[START]`, `[STEP]`, and `[END]` lines to stdout so it
matches the hackathon submission parser.

## Validation

```bash
./venv/bin/openenv validate
```

## Project Layout

```text
.
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── rag_env/
├── server/
│   ├── app.py
│   ├── environment.py
│   └── requirements.txt
└── tasks.py
```
