"""
server/environment.py
======================
The core environment logic.
Inherits from openenv-core's Environment base class.

This is the BRAIN:
- Simulated internal company database (with correct + outdated docs)
- reset() → fresh episode
- step()  → agent acts, environment responds
- state   → current episode state (property)
"""

import uuid
from typing import Optional
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import RAGAction, RAGObservation, RAGState


# ─────────────────────────────────────────────
# Internal Knowledge Base (Fake Company DB)
# ─────────────────────────────────────────────
class InternalDatabase:
    """
    Simulates a real company's internal knowledge base.
    Has CORRECT documents AND some intentionally outdated ones.
    This is what causes hallucinations in real RAG systems.
    """

    def __init__(self):
        self.documents = self._load_documents()
        self.fix_log: list[dict] = []

    def _load_documents(self) -> dict:
        return {
            # ── LEAVE POLICY (3 versions, only v3 is correct) ──
            "hr_001": {
                "id": "hr_001", "title": "Leave Policy v1 (2020)",
                "content": "Employees are entitled to 10 days of annual leave per year.",
                "date": "2020-01-01", "version": 1, "topic": "leave_policy",
                "is_outdated": True, "correct_doc_id": "hr_003"
            },
            "hr_002": {
                "id": "hr_002", "title": "Leave Policy v2 (2022)",
                "content": "Employees are entitled to 15 days of annual leave per year.",
                "date": "2022-06-01", "version": 2, "topic": "leave_policy",
                "is_outdated": True, "correct_doc_id": "hr_003"
            },
            "hr_003": {
                "id": "hr_003", "title": "Leave Policy v3 (2024) - CURRENT",
                "content": "Employees are entitled to 20 days of annual leave per year. Effective January 2024.",
                "date": "2024-01-01", "version": 3, "topic": "leave_policy",
                "is_outdated": False, "correct_doc_id": None
            },

            # ── PRICING (3 versions, only v3 is correct) ──
            "prod_001": {
                "id": "prod_001", "title": "Pricing 2021",
                "content": "Our standard plan costs $50 per month.",
                "date": "2021-03-01", "version": 1, "topic": "pricing",
                "is_outdated": True, "correct_doc_id": "prod_003"
            },
            "prod_002": {
                "id": "prod_002", "title": "Pricing 2023",
                "content": "Our standard plan costs $75 per month.",
                "date": "2023-01-01", "version": 2, "topic": "pricing",
                "is_outdated": True, "correct_doc_id": "prod_003"
            },
            "prod_003": {
                "id": "prod_003", "title": "Pricing 2024 - CURRENT",
                "content": "Our standard plan costs $99 per month. Premium plan is $199 per month.",
                "date": "2024-06-01", "version": 3, "topic": "pricing",
                "is_outdated": False, "correct_doc_id": None
            },

            # ── REFUND POLICY (2 versions) ──
            "ref_001": {
                "id": "ref_001", "title": "Refund Policy 2022",
                "content": "Customers can request a refund within 7 days of purchase.",
                "date": "2022-05-01", "version": 1, "topic": "refund_policy",
                "is_outdated": True, "correct_doc_id": "ref_002"
            },
            "ref_002": {
                "id": "ref_002", "title": "Refund Policy 2024 - CURRENT",
                "content": "Customers can request a refund within 30 days of purchase. No questions asked.",
                "date": "2024-02-01", "version": 2, "topic": "refund_policy",
                "is_outdated": False, "correct_doc_id": None
            },
        }

    def search(self, topic: str) -> list[dict]:
        results = [
            doc for doc in self.documents.values()
            if topic.lower() in doc["topic"].lower()
            and not doc.get("archived", False)
        ]
        results.sort(key=lambda x: x["date"], reverse=True)
        return results

    def get_all_versions(self, topic: str) -> list[dict]:
        return [d for d in self.documents.values() if d["topic"] == topic]

    def get_latest_version(self, topic: str) -> Optional[dict]:
        docs = self.get_all_versions(topic)
        return max(docs, key=lambda x: x["date"]) if docs else None

    def fix_document(self, wrong_doc_id: str) -> dict:
        """Auto-fix: archive wrong doc and promote the correct version."""
        if wrong_doc_id not in self.documents:
            return {"success": False, "message": "Document not found"}

        doc = self.documents[wrong_doc_id]
        correct_id = doc.get("correct_doc_id")

        if not correct_id:
            return {"success": False, "message": "No newer version found internally"}

        self.documents[wrong_doc_id]["archived"] = True
        self.documents[wrong_doc_id]["archive_reason"] = "Auto-fixed: outdated"

        fix_record = {
            "archived": wrong_doc_id,
            "promoted": correct_id,
            "reason": "Hallucination detected — outdated document replaced",
        }
        self.fix_log.append(fix_record)

        return {
            "success": True,
            "message": f"Archived '{doc['title']}', promoted '{self.documents[correct_id]['title']}'",
            "correct_doc": self.documents[correct_id],
        }


# ─────────────────────────────────────────────
# Tasks (3 difficulty levels)
# ─────────────────────────────────────────────
TASKS = {
    "task_detect_hallucination": {
        "question": "How many days of annual leave do employees get?",
        "topic": "leave_policy",
        "correct_answer": "20 days",
        "difficulty": "easy",
        "max_steps": 4,
        "passing_score": 0.6,
    },
    "task_find_source": {
        "question": "What is the current price of the standard plan?",
        "topic": "pricing",
        "correct_answer": "$99 per month",
        "wrong_doc_id": "prod_001",
        "difficulty": "medium",
        "max_steps": 6,
        "passing_score": 0.7,
    },
    "task_full_pipeline": {
        "question": "What is our refund policy?",
        "topic": "refund_policy",
        "correct_answer": "30 days",
        "wrong_doc_id": "ref_001",
        "difficulty": "hard",
        "max_steps": 10,
        "passing_score": 0.85,
    },
}


# ─────────────────────────────────────────────
# Main Environment Class
# ─────────────────────────────────────────────
class RAGEnvironment(Environment):
    """
    Self-Healing RAG Environment.
    Inherits from openenv-core Environment base class.
    """

    def __init__(self):
        super().__init__()
        self._state = RAGState()
        self._db = InternalDatabase()
        self._task: dict = {}
        self._ctx: dict = {}   # current episode context

    # ── reset ──────────────────────────────────
    def reset(self, task_name: str = "task_detect_hallucination") -> RAGObservation:
        """Start a fresh episode for the given task."""
        self._db = InternalDatabase()
        self._task = TASKS.get(task_name, TASKS["task_detect_hallucination"])

        self._state = RAGState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task=task_name,
            hallucination_detected=False,
            database_fixed=False,
            fix_log=[],
            episode_rewards=[],
            done=False,
        )

        self._ctx = {
            "hallucination_detected": False,
            "database_fixed": False,
            "current_answer": None,
            "conflicting_docs": [],
        }

        docs = self._db.search(self._task["topic"])

        return RAGObservation(
            question=self._task["question"],
            retrieved_documents=docs,
            step_number=0,
            message="Episode started. Analyze the documents and answer the question.",
            reward=0.0,
        )

    # ── step ───────────────────────────────────
    def step(self, action: RAGAction) -> RAGObservation:
        """Agent takes an action; environment responds with observation + reward."""
        self._state.step_count += 1

        if action.action_type == "answer":
            obs = self._handle_answer(action)
        elif action.action_type == "detect":
            obs = self._handle_detect(action)
        elif action.action_type == "find_source":
            obs = self._handle_find_source(action)
        elif action.action_type == "fix":
            obs = self._handle_fix(action)
        elif action.action_type == "verify":
            obs = self._handle_verify(action)
        else:
            obs = self._make_obs("Unknown action type.", reward=0.0)

        self._state.episode_rewards.append(obs.reward)
        self._state.hallucination_detected = self._ctx["hallucination_detected"]
        self._state.database_fixed = self._ctx["database_fixed"]
        self._state.fix_log = self._db.fix_log

        # Mark done if max steps reached or verify succeeded
        if self._state.step_count >= self._task["max_steps"]:
            self._state.done = True

        return obs

    # ── state (property) ───────────────────────
    @property
    def state(self) -> RAGState:
        return self._state

    # ── Action Handlers ────────────────────────

    def _handle_answer(self, action: RAGAction) -> RAGObservation:
        self._ctx["current_answer"] = action.content
        correct = self._task["correct_answer"].lower()
        given = action.content.lower()

        if correct in given or given in correct:
            return self._make_obs(
                "Answer correct! Now check for hallucination risk in the documents.",
                reward=0.6,
            )
        else:
            self._ctx["hallucination_detected"] = True
            return self._make_obs(
                "Wrong answer — likely from an outdated document. Run detect to confirm.",
                reward=0.1,
            )

    def _handle_detect(self, action: RAGAction) -> RAGObservation:
        all_vers = self._db.get_all_versions(self._task["topic"])
        outdated = [d for d in all_vers if d.get("is_outdated")]

        if outdated:
            self._ctx["conflicting_docs"] = outdated
            self._ctx["hallucination_detected"] = True
            return self._make_obs(
                f"Detected {len(outdated)} outdated document(s). Find the source.",
                reward=0.7,
            )
        return self._make_obs("No conflicts found. Look more carefully.", reward=0.2)

    def _handle_find_source(self, action: RAGAction) -> RAGObservation:
        all_vers = self._db.get_all_versions(self._task["topic"])
        outdated_ids = [d["id"] for d in all_vers if d.get("is_outdated")]
        known_wrong = self._task.get("wrong_doc_id", "")

        if action.target_doc_id == known_wrong:
            return self._make_obs(
                f"Correct source found: {action.target_doc_id}. Now fix the database.",
                reward=0.8,
            )
        elif action.target_doc_id in outdated_ids:
            return self._make_obs(
                "Partially correct — that document is outdated but not the primary source.",
                reward=0.4,
            )
        return self._make_obs("Wrong document identified. Try again.", reward=0.2)

    def _handle_fix(self, action: RAGAction) -> RAGObservation:
        if not action.target_doc_id:
            return self._make_obs("Specify target_doc_id to fix.", reward=0.0)

        result = self._db.fix_document(action.target_doc_id)

        if result["success"]:
            self._ctx["database_fixed"] = True
            return self._make_obs(
                f"Database fixed! {result['message']}. Now verify with correct answer.",
                reward=0.9,
            )
        return self._make_obs(f"Fix failed: {result['message']}", reward=0.2)

    def _handle_verify(self, action: RAGAction) -> RAGObservation:
        if not self._ctx["database_fixed"]:
            return self._make_obs("Fix the database before verifying.", reward=0.0)

        correct = self._task["correct_answer"].lower()
        given = action.content.lower()

        if correct in given or given in correct:
            self._state.done = True
            return self._make_obs(
                "COMPLETE! Full self-healing pipeline succeeded!",
                reward=1.0,
            )
        return self._make_obs(
            "Fix applied but answer still incorrect. Check the latest document.",
            reward=0.5,
        )

    # ── Helper ─────────────────────────────────
    def _make_obs(self, message: str, reward: float) -> RAGObservation:
        docs = self._db.search(self._task["topic"])
        return RAGObservation(
            question=self._task.get("question", ""),
            retrieved_documents=docs,
            current_answer=self._ctx.get("current_answer"),
            hallucination_detected=self._ctx["hallucination_detected"],
            conflicting_docs=self._ctx["conflicting_docs"],
            database_fixed=self._ctx["database_fixed"],
            step_number=self._state.step_count,
            message=message,
            reward=reward,
        )
