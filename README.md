---
title: Self-Healing RAG Environment
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - rag
  - hallucination
  - self-healing
---

# Self-Healing RAG Environment 🔧

> An OpenEnv environment where AI agents detect and automatically fix hallucinations
> in a company's internal knowledge base — using only internal document cross-referencing.

---

## The Problem This Solves

```
Company internal database → has outdated documents
RAG AI searches database → finds outdated doc
AI gives wrong answer    → hallucination
Nobody knows which doc   → caused it
No automatic fix exists  → it keeps happening
```

**This environment trains and evaluates agents that solve this entire pipeline.**

---

## How It Works

```
1. Agent receives question + internal documents
2. Agent gives initial answer
3. Agent detects conflicting/outdated documents
4. Agent identifies which specific document caused the hallucination
5. Agent auto-fixes the database (archives wrong doc, promotes latest)
6. Agent verifies the corrected answer
```

All using **only internal documents** — no internet, no external APIs.

---

## Quick Start

```bash
# Install client
pip install git+https://huggingface.co/spaces/YOUR_USERNAME/self-healing-rag

# Use it
from rag_env import RAGEnv, RAGAction

with RAGEnv(base_url="https://YOUR_USERNAME-self-healing-rag.hf.space").sync() as env:
    obs = env.reset(task_name="task_detect_hallucination")
    result = env.step(RAGAction(action_type="answer", content="20 days"))
    print(result.observation.message)
```

---

## Tasks

| Task | Difficulty | Description | Passing Score |
|------|-----------|-------------|---------------|
| `task_detect_hallucination` | Easy | Detect if hallucination exists in retrieved docs | 0.6 |
| `task_find_source` | Medium | Find which specific document caused the hallucination | 0.7 |
| `task_full_pipeline` | Hard | Full pipeline: detect → find → fix → verify | 0.85 |

---

## Action Space

| Action | What it does | Required fields |
|--------|-------------|-----------------|
| `answer` | Give answer to question | `content` |
| `detect` | Scan docs for conflicts | `content` |
| `find_source` | Identify wrong document | `content`, `target_doc_id` |
| `fix` | Auto-fix database | `target_doc_id` |
| `verify` | Confirm fix with correct answer | `content` |

---

## Observation Space

```python
RAGObservation(
    question: str,
    retrieved_documents: list[dict],
    current_answer: str | None,
    hallucination_detected: bool,
    conflicting_docs: list[dict],
    database_fixed: bool,
    step_number: int,
    message: str,
    reward: float          # step-level reward 0.0–1.0
)
```

---

## Reward Function

Rewards given at every step (not just at end):

| Step | Action | Reward | Why |
|------|--------|--------|-----|
| 1 | Correct answer | 0.6 | Good start |
| 2 | Hallucination detected | 0.7 | Found the problem |
| 3 | Source doc identified | 0.8 | Found the cause |
| 4 | Database fixed | 0.9 | Fixed the cause |
| 5 | Verified correct | 1.0 | Complete pipeline |

---

## Run Inference

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=https://YOUR_USERNAME-self-healing-rag.hf.space

python inference.py
```

---

## Run with Docker

```bash
docker build -f server/Dockerfile -t self-healing-rag .
docker run -p 7860:7860 self-healing-rag
```

---

## Validate

```bash
pip install openenv-core
openenv validate
```

---

## Project Structure

```
self-healing-rag/
├── models.py           # RAGAction, RAGObservation, RAGState (Pydantic)
├── client.py           # RAGEnv HTTP client (openenv-core)
├── __init__.py         # Package exports
├── inference.py        # Baseline inference script (MANDATORY)
├── openenv.yaml        # OpenEnv manifest (MANDATORY)
├── pyproject.toml      # pip-installable package
├── README.md           # This file
└── server/
    ├── app.py          # FastAPI server (openenv-core)
    ├── environment.py  # Core logic (inherits Environment)
    ├── requirements.txt
    ├── Dockerfile
    └── __init__.py
```

---

## Baseline Scores

| Task | Difficulty | Score | Steps |
|------|-----------|-------|-------|
| task_detect_hallucination | Easy | 0.70 | 2 |
| task_find_source | Medium | 0.80 | 3 |
| task_full_pipeline | Hard | 0.90 | 5 |

---

Built for the OpenEnv Hackathon — judged by Hugging Face & Meta engineers.
