from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List


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
        "description": (
            "Detect when a response was grounded in a superseded internal document."
        ),
        "difficulty": "easy",
        "max_steps": 4,
        "passing_score": 0.6,
    },
    "task_find_source": {
        "description": (
            "Identify which outdated document caused the hallucinated answer."
        ),
        "difficulty": "medium",
        "max_steps": 6,
        "passing_score": 0.7,
    },
    "task_full_pipeline": {
        "description": (
            "Run the full self-healing loop: answer, detect, find, fix, and verify."
        ),
        "difficulty": "hard",
        "max_steps": 10,
        "passing_score": 0.85,
    },
    "task_cross_topic_audit": {
        "description": (
            "Audit multiple topics, archive every outdated document, and confirm the KB is healed."
        ),
        "difficulty": "expert",
        "max_steps": 15,
        "passing_score": 0.9,
    },
}


SCENARIO_BANK: Dict[str, Dict[str, Any]] = {
    "leave_policy": {
        "topic": "leave_policy",
        "question": "How many days of annual leave do full-time employees receive?",
        "canonical_answer": "20 days",
        "answer_aliases": [
            "20 days",
            "20 days of annual leave",
            "20 days per year",
        ],
        "current_template_id": "leave_v3",
        "documents": [
            {
                "template_id": "leave_v1",
                "title": "Employee Leave Policy Update - 2020",
                "content": (
                    "Full-time employees receive 10 days of annual leave per "
                    "calendar year."
                ),
                "date": "2020-01-01",
                "topic": "leave_policy",
                "answer_value": "10 days",
                "is_outdated": True,
                "correct_template_id": "leave_v3",
            },
            {
                "template_id": "leave_v2",
                "title": "Employee Leave Policy Update - 2022",
                "content": (
                    "Full-time employees receive 15 days of annual leave per "
                    "calendar year."
                ),
                "date": "2022-06-01",
                "topic": "leave_policy",
                "answer_value": "15 days",
                "is_outdated": True,
                "correct_template_id": "leave_v3",
            },
            {
                "template_id": "leave_v3",
                "title": "Employee Leave Policy Update - 2024",
                "content": (
                    "Effective January 2024, full-time employees receive 20 "
                    "days of annual leave per calendar year."
                ),
                "date": "2024-01-01",
                "topic": "leave_policy",
                "answer_value": "20 days",
                "is_outdated": False,
                "correct_template_id": None,
            },
            {
                "template_id": "leave_handbook",
                "title": "People Handbook Notes",
                "content": (
                    "Always cite the most recently approved policy revision when "
                    "answering employee leave questions."
                ),
                "date": "2024-02-10",
                "topic": "leave_policy",
                "answer_value": None,
                "is_outdated": False,
                "correct_template_id": None,
            },
        ],
    },
    "pricing_standard": {
        "topic": "pricing",
        "question": "What is the current monthly price of the Standard plan?",
        "canonical_answer": "$99 per month",
        "answer_aliases": [
            "$99 per month",
            "$99 monthly",
            "99 dollars per month",
        ],
        "current_template_id": "pricing_v3",
        "documents": [
            {
                "template_id": "pricing_v1",
                "title": "Standard Plan Pricing - 2021",
                "content": "The Standard plan costs $49 per month.",
                "date": "2021-03-01",
                "topic": "pricing",
                "answer_value": "$49 per month",
                "is_outdated": True,
                "correct_template_id": "pricing_v3",
            },
            {
                "template_id": "pricing_v2",
                "title": "Standard Plan Pricing - 2023",
                "content": "The Standard plan costs $79 per month.",
                "date": "2023-01-01",
                "topic": "pricing",
                "answer_value": "$79 per month",
                "is_outdated": True,
                "correct_template_id": "pricing_v3",
            },
            {
                "template_id": "pricing_v3",
                "title": "Standard Plan Pricing - 2024",
                "content": (
                    "As of June 2024, the Standard plan costs $99 per month."
                ),
                "date": "2024-06-01",
                "topic": "pricing",
                "answer_value": "$99 per month",
                "is_outdated": False,
                "correct_template_id": None,
            },
            {
                "template_id": "pricing_guide",
                "title": "Packaging Guide",
                "content": (
                    "Legacy pricing screenshots may still exist in internal decks; "
                    "confirm against the latest pricing bulletin."
                ),
                "date": "2024-06-15",
                "topic": "pricing",
                "answer_value": None,
                "is_outdated": False,
                "correct_template_id": None,
            },
        ],
    },
    "refund_policy": {
        "topic": "refund_policy",
        "question": "Within how many days can a customer request a refund?",
        "canonical_answer": "30 days",
        "answer_aliases": [
            "30 days",
            "within 30 days",
            "30-day refund window",
        ],
        "current_template_id": "refund_v3",
        "documents": [
            {
                "template_id": "refund_v1",
                "title": "Refund Policy Memo - 2021",
                "content": "Customers may request a refund within 7 days of purchase.",
                "date": "2021-04-01",
                "topic": "refund_policy",
                "answer_value": "7 days",
                "is_outdated": True,
                "correct_template_id": "refund_v3",
            },
            {
                "template_id": "refund_v2",
                "title": "Refund Policy Memo - 2023",
                "content": "Customers may request a refund within 14 days of purchase.",
                "date": "2023-01-10",
                "topic": "refund_policy",
                "answer_value": "14 days",
                "is_outdated": True,
                "correct_template_id": "refund_v3",
            },
            {
                "template_id": "refund_v3",
                "title": "Refund Policy Memo - 2024",
                "content": (
                    "Starting February 2024, customers may request a refund within "
                    "30 days of purchase."
                ),
                "date": "2024-02-01",
                "topic": "refund_policy",
                "answer_value": "30 days",
                "is_outdated": False,
                "correct_template_id": None,
            },
            {
                "template_id": "refund_playbook",
                "title": "CX Escalation Playbook",
                "content": (
                    "Escalation agents should verify the active refund window before "
                    "responding to chargeback threats."
                ),
                "date": "2024-02-14",
                "topic": "refund_policy",
                "answer_value": None,
                "is_outdated": False,
                "correct_template_id": None,
            },
        ],
    },
    "travel_budget": {
        "topic": "travel_budget",
        "question": (
            "What is the quarterly travel reimbursement cap for field engineers?"
        ),
        "canonical_answer": "$1,200 per quarter",
        "answer_aliases": [
            "$1,200 per quarter",
            "$1200 per quarter",
            "1200 dollars per quarter",
        ],
        "current_template_id": "travel_v3",
        "documents": [
            {
                "template_id": "travel_v1",
                "title": "Field Travel Budget - 2021",
                "content": (
                    "Field engineers may claim up to $600 per quarter in travel "
                    "reimbursements."
                ),
                "date": "2021-05-01",
                "topic": "travel_budget",
                "answer_value": "$600 per quarter",
                "is_outdated": True,
                "correct_template_id": "travel_v3",
            },
            {
                "template_id": "travel_v2",
                "title": "Field Travel Budget - 2023",
                "content": (
                    "Field engineers may claim up to $900 per quarter in travel "
                    "reimbursements."
                ),
                "date": "2023-05-01",
                "topic": "travel_budget",
                "answer_value": "$900 per quarter",
                "is_outdated": True,
                "correct_template_id": "travel_v3",
            },
            {
                "template_id": "travel_v3",
                "title": "Field Travel Budget - 2024",
                "content": (
                    "Field engineers may claim up to $1,200 per quarter in travel "
                    "reimbursements."
                ),
                "date": "2024-03-15",
                "topic": "travel_budget",
                "answer_value": "$1,200 per quarter",
                "is_outdated": False,
                "correct_template_id": None,
            },
            {
                "template_id": "travel_checklist",
                "title": "Expense Review Checklist",
                "content": (
                    "Reviewers should reject reimbursement answers that cite an "
                    "archived travel budget notice."
                ),
                "date": "2024-03-20",
                "topic": "travel_budget",
                "answer_value": None,
                "is_outdated": False,
                "correct_template_id": None,
            },
        ],
    },
    "support_sla": {
        "topic": "support_sla",
        "question": (
            "What is the first-response SLA for enterprise support tickets?"
        ),
        "canonical_answer": "12 hours",
        "answer_aliases": [
            "12 hours",
            "within 12 hours",
            "12-hour SLA",
        ],
        "current_template_id": "sla_v3",
        "documents": [
            {
                "template_id": "sla_v1",
                "title": "Enterprise Support SLA - 2020",
                "content": (
                    "Enterprise support tickets receive a first response within "
                    "48 hours."
                ),
                "date": "2020-09-01",
                "topic": "support_sla",
                "answer_value": "48 hours",
                "is_outdated": True,
                "correct_template_id": "sla_v3",
            },
            {
                "template_id": "sla_v2",
                "title": "Enterprise Support SLA - 2022",
                "content": (
                    "Enterprise support tickets receive a first response within "
                    "24 hours."
                ),
                "date": "2022-08-15",
                "topic": "support_sla",
                "answer_value": "24 hours",
                "is_outdated": True,
                "correct_template_id": "sla_v3",
            },
            {
                "template_id": "sla_v3",
                "title": "Enterprise Support SLA - 2024",
                "content": (
                    "Effective 2024, enterprise support tickets receive a first "
                    "response within 12 hours."
                ),
                "date": "2024-01-20",
                "topic": "support_sla",
                "answer_value": "12 hours",
                "is_outdated": False,
                "correct_template_id": None,
            },
            {
                "template_id": "sla_runbook",
                "title": "Support Escalation Runbook",
                "content": (
                    "Priority response expectations change over time; check the "
                    "newest signed SLA notice."
                ),
                "date": "2024-02-01",
                "topic": "support_sla",
                "answer_value": None,
                "is_outdated": False,
                "correct_template_id": None,
            },
        ],
    },
    "remote_work": {
        "topic": "remote_work",
        "question": "How many remote work days per week are employees allowed?",
        "canonical_answer": "4 days per week",
        "answer_aliases": [
            "4 days per week",
            "4 remote days per week",
            "4 days",
        ],
        "current_template_id": "remote_v3",
        "documents": [
            {
                "template_id": "remote_v1",
                "title": "Remote Work Policy - 2021",
                "content": "Employees may work remotely 2 days per week.",
                "date": "2021-11-01",
                "topic": "remote_work",
                "answer_value": "2 days per week",
                "is_outdated": True,
                "correct_template_id": "remote_v3",
            },
            {
                "template_id": "remote_v2",
                "title": "Remote Work Policy - 2023",
                "content": "Employees may work remotely 3 days per week.",
                "date": "2023-07-01",
                "topic": "remote_work",
                "answer_value": "3 days per week",
                "is_outdated": True,
                "correct_template_id": "remote_v3",
            },
            {
                "template_id": "remote_v3",
                "title": "Remote Work Policy - 2024",
                "content": (
                    "Employees may work remotely 4 days per week unless their role "
                    "requires additional onsite coverage."
                ),
                "date": "2024-04-01",
                "topic": "remote_work",
                "answer_value": "4 days per week",
                "is_outdated": False,
                "correct_template_id": None,
            },
            {
                "template_id": "remote_manager_guide",
                "title": "Manager Scheduling Guide",
                "content": (
                    "Team schedules should reference the current remote work policy, "
                    "not archived rollout memos."
                ),
                "date": "2024-04-08",
                "topic": "remote_work",
                "answer_value": None,
                "is_outdated": False,
                "correct_template_id": None,
            },
        ],
    },
}


TASK_SCENARIO_POOLS: Dict[str, List[str]] = {
    "task_detect_hallucination": [
        "leave_policy",
        "pricing_standard",
        "refund_policy",
        "remote_work",
    ],
    "task_find_source": [
        "pricing_standard",
        "refund_policy",
        "travel_budget",
        "support_sla",
        "remote_work",
    ],
    "task_full_pipeline": list(SCENARIO_BANK.keys()),
    "task_cross_topic_audit": list(SCENARIO_BANK.keys()),
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


def build_task_instance(task_id: str, seed: int | str | None = None) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Task {task_id} not found")

    rng = random.Random(str(seed) if seed is not None else None)
    task = copy.deepcopy(TASKS[task_id])

    if task_id == "task_cross_topic_audit":
        scenario_ids = TASK_SCENARIO_POOLS[task_id]
        selected_ids = rng.sample(scenario_ids, k=min(3, len(scenario_ids)))
        documents: List[Dict[str, Any]] = []
        selected_topics: List[str] = []
        current_answers: Dict[str, str] = {}
        total_outdated = 0

        for scenario_id in selected_ids:
            materialized = _materialize_scenario(scenario_id, rng)
            documents.extend(materialized["documents"])
            selected_topics.append(materialized["topic"])
            current_answers[materialized["topic"]] = materialized["canonical_answer"]
            total_outdated += len(materialized["outdated_doc_ids"])

        rng.shuffle(documents)
        return {
            "task_name": task_id,
            "question": (
                "Audit the knowledge base across multiple topics. Find every "
                "superseded document, archive it, and confirm the current answers."
            ),
            "topic": "all",
            "documents": documents,
            "selected_topics": selected_topics,
            "current_answers": current_answers,
            "total_outdated": total_outdated,
            **task,
        }

    scenario_id = rng.choice(TASK_SCENARIO_POOLS[task_id])
    materialized = _materialize_scenario(scenario_id, rng)
    return {
        "task_name": task_id,
        "question": materialized["question"],
        "topic": materialized["topic"],
        "documents": materialized["documents"],
        "current_doc_id": materialized["current_doc_id"],
        "correct_answer": materialized["canonical_answer"],
        "answer_aliases": materialized["answer_aliases"],
        "scenario_id": scenario_id,
        **task,
    }


def _materialize_scenario(
    scenario_id: str, rng: random.Random
) -> Dict[str, Any]:
    scenario = copy.deepcopy(SCENARIO_BANK[scenario_id])
    suffix = f"{rng.randrange(16**6):06x}"
    template_ids = [doc["template_id"] for doc in scenario["documents"]]
    id_map = {
        template_id: f"{scenario_id}_{index + 1}_{suffix}"
        for index, template_id in enumerate(template_ids)
    }

    documents: List[Dict[str, Any]] = []
    outdated_doc_ids: List[str] = []
    for template in scenario["documents"]:
        materialized = {
            "id": id_map[template["template_id"]],
            "title": template["title"],
            "content": template["content"],
            "date": template["date"],
            "topic": template["topic"],
            "answer_value": template.get("answer_value"),
            "is_outdated": bool(template.get("is_outdated", False)),
            "correct_doc_id": (
                id_map[template["correct_template_id"]]
                if template.get("correct_template_id")
                else None
            ),
            "archived": False,
        }
        documents.append(materialized)
        if materialized["is_outdated"]:
            outdated_doc_ids.append(materialized["id"])

    rng.shuffle(documents)
    return {
        "topic": scenario["topic"],
        "question": scenario["question"],
        "canonical_answer": scenario["canonical_answer"],
        "answer_aliases": scenario["answer_aliases"],
        "current_doc_id": id_map[scenario["current_template_id"]],
        "outdated_doc_ids": outdated_doc_ids,
        "documents": documents,
    }
