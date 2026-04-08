from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TaskDefinition:
    id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    passing_score: float


TASKS: Dict[str, Dict[str, object]] = {
    "task_detect_hallucination": {
        "question": "How many days of annual leave do employees get?",
        "topic": "leave_policy",
        "correct_answer": "20 days",
        "difficulty": "easy",
        "max_steps": 4,
        "passing_score": 0.6,
        "description": "Detect if the AI answer contains hallucination from outdated internal documents",
    },
    "task_find_source": {
        "question": "What is the current price of the standard plan?",
        "topic": "pricing",
        "correct_answer": "$99 per month",
        "wrong_doc_id": "prod_001",
        "difficulty": "medium",
        "max_steps": 6,
        "passing_score": 0.7,
        "description": "Find which specific internal document is causing the hallucination",
    },
    "task_full_pipeline": {
        "question": "What is our refund policy?",
        "topic": "refund_policy",
        "correct_answer": "30 days",
        "wrong_doc_id": "ref_001",
        "difficulty": "hard",
        "max_steps": 10,
        "passing_score": 0.85,
        "description": "Full self-healing pipeline detect, find source, fix database, verify",
    },
    "task_cross_topic_audit": {
        "question": "Audit the entire knowledge base. Find and fix ALL outdated documents across ALL topics.",
        "topic": "all",
        "correct_answer": "audit_complete",
        "difficulty": "expert",
        "max_steps": 15,
        "passing_score": 0.9,
        "total_outdated": 5,
        "description": "Expert autonomous audit of entire knowledge base across all topics",
    },
}


def list_tasks() -> List[TaskDefinition]:
    return [
        TaskDefinition(
            id=task_id,
            name=task_id,
            description=str(task["description"]),
            difficulty=str(task["difficulty"]),
            max_steps=int(task["max_steps"]),
            passing_score=float(task["passing_score"]),
        )
        for task_id, task in TASKS.items()
    ]


def get_task_by_id(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise ValueError(f"Task {task_id} not found")
    task = TASKS[task_id]
    return TaskDefinition(
        id=task_id,
        name=task_id,
        description=str(task["description"]),
        difficulty=str(task["difficulty"]),
        max_steps=int(task["max_steps"]),
        passing_score=float(task["passing_score"]),
    )
