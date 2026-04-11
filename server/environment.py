from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from models import RAGAction, RAGObservation, RAGState
    from tasks import TASKS, build_task_instance
except ImportError:
    from ..models import RAGAction, RAGObservation, RAGState
    from ..tasks import TASKS, build_task_instance


MIN_SCORE = 0.01
MAX_SCORE = 0.99
CONFLICT_KEYWORDS = (
    "conflict",
    "contradict",
    "outdated",
    "stale",
    "older",
    "superseded",
)


class InternalDatabase:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = {doc["id"]: dict(doc) for doc in documents}
        self.visible_order = [doc["id"] for doc in documents]
        self.fix_log: List[Dict[str, str]] = []

    def search(self, topic: str) -> List[Dict[str, Any]]:
        return [
            self._visible_document(self.documents[doc_id])
            for doc_id in self.visible_order
            if doc_id in self.documents
            and not self.documents[doc_id].get("archived", False)
            and (topic == "all" or self.documents[doc_id]["topic"] == topic)
        ]

    def get_document(self, doc_id: str | None) -> Dict[str, Any] | None:
        if not doc_id:
            return None
        return self.documents.get(doc_id)

    def get_versions(self, topics: List[str] | None = None) -> List[Dict[str, Any]]:
        allowed_topics = set(topics) if topics else None
        return [
            self.documents[doc_id]
            for doc_id in self.visible_order
            if doc_id in self.documents
            and not self.documents[doc_id].get("archived", False)
            and (
                allowed_topics is None
                or self.documents[doc_id]["topic"] in allowed_topics
            )
        ]

    def get_all_outdated(self, topics: List[str] | None = None) -> List[Dict[str, Any]]:
        return [doc for doc in self.get_versions(topics) if doc.get("is_outdated")]

    def get_all_topics_with_conflicts(
        self, topics: List[str] | None = None
    ) -> Dict[str, List[str]]:
        conflicts: Dict[str, List[str]] = {}
        for doc in self.get_all_outdated(topics):
            conflicts.setdefault(doc["topic"], []).append(doc["id"])
        return conflicts

    def count_remaining_outdated(self, topics: List[str] | None = None) -> int:
        return len(self.get_all_outdated(topics))

    def fix_document(self, doc_id: str) -> Dict[str, Any]:
        doc = self.documents.get(doc_id)
        if doc is None:
            return {"success": False, "message": "Document not found"}
        if doc.get("archived", False):
            return {"success": False, "message": "Document already archived"}
        if not doc.get("is_outdated", False):
            return {"success": False, "message": "Document is already current"}

        self.documents[doc_id]["archived"] = True
        promoted_id = doc.get("correct_doc_id")
        self.fix_log.append({"archived": doc_id, "promoted": promoted_id or ""})
        return {
            "success": True,
            "message": f"Archived {doc_id}",
            "correct_doc": self.documents.get(promoted_id),
        }

    @staticmethod
    def _visible_document(doc: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "date": doc["date"],
            "topic": doc["topic"],
        }


class RAGEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._task_name = "task_detect_hallucination"
        self._task = build_task_instance(self._task_name)
        self._db = InternalDatabase(self._task["documents"])
        self._ctx = self._fresh_context()
        self._step_count = 0
        self._episode_id = ""
        self._rewards: List[float] = []
        self._done = False
        self._actions: List[Dict[str, Any]] = []

    def _fresh_context(self) -> Dict[str, Any]:
        return {
            "hallucination_detected": False,
            "database_fixed": False,
            "current_answer": None,
            "conflicting_doc_ids": [],
            "fixes_applied": 0,
            "source_found": False,
            "source_doc_id": None,
            "matched_answer_doc_id": None,
            "answer_matches_known_doc": False,
            "answer_is_current": False,
            "verified_correct": False,
            "audit_completed": False,
            "detection_done": False,
        }

    def reset(self, **kwargs):
        task_name = kwargs.get("task_name", "task_detect_hallucination")
        if task_name not in TASKS:
            task_name = "task_detect_hallucination"

        self._task_name = task_name
        self._task = build_task_instance(task_name, seed=kwargs.get("seed"))
        self._db = InternalDatabase(self._task["documents"])
        self._ctx = self._fresh_context()
        self._step_count = 0
        self._episode_id = str(uuid.uuid4())
        self._rewards = []
        self._done = False
        self._actions = []

        if task_name == "task_cross_topic_audit":
            topics = ", ".join(self._task["selected_topics"])
            message = (
                "Audit started. Review all supplied topics, archive every stale "
                f"document, then confirm the KB is healed. Topics: {topics}."
            )
        else:
            message = (
                "Episode started. Inspect the retrieved documents, ground the "
                "answer in the newest evidence, and repair any stale source."
            )

        return RAGObservation(
            question=self._task["question"],
            retrieved_documents=self._db.search(self._task.get("topic", "all")),
            step_number=0,
            message=message,
            reward=self._clamp_reward(MIN_SCORE),
            done=False,
        )

    def step(self, action: RAGAction):
        if self._done:
            return self._obs("Episode already finished. Reset to start again.", MIN_SCORE)

        self._step_count += 1
        self._actions.append(
            {
                "action_type": action.action_type,
                "content": action.content,
                "target_doc_id": action.target_doc_id,
            }
        )

        if self._task_name == "task_cross_topic_audit":
            obs = self._audit_step(action)
        else:
            obs = self._single_topic_step(action)

        self._rewards.append(obs.reward)
        if self._step_count >= int(self._task["max_steps"]) and not self._done:
            self._done = True
            obs.done = True
            if "step limit reached" not in obs.message.lower():
                obs.message = f"{obs.message} Step limit reached."
        return obs

    @property
    def state(self):
        return RAGState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            current_task=self._task_name,
            hallucination_detected=self._ctx["hallucination_detected"],
            database_fixed=self._ctx["database_fixed"],
            fix_log=self._db.fix_log,
            episode_rewards=self._rewards,
            done=self._done,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="self-healing-rag",
            description=(
                "Self-healing RAG benchmark with randomized task instances and "
                "hidden ground-truth document labels."
            ),
            version="1.1.0",
            author="Pradeep K",
        )

    def _single_topic_step(self, action: RAGAction) -> RAGObservation:
        if action.action_type == "answer":
            return self._answer(action)
        if action.action_type == "detect":
            return self._detect(action)
        if action.action_type == "find_source":
            return self._find(action)
        if action.action_type == "fix":
            return self._fix(action)
        if action.action_type == "verify":
            return self._verify(action)
        return self._obs("Unknown action. Use answer, detect, find_source, fix, or verify.", MIN_SCORE)

    def _audit_step(self, action: RAGAction) -> RAGObservation:
        if action.action_type == "detect":
            return self._audit_detect(action)
        if action.action_type == "find_source":
            return self._audit_find(action)
        if action.action_type == "fix":
            return self._audit_fix(action)
        if action.action_type == "verify":
            return self._audit_verify(action)
        return self._obs(
            "Use detect, find_source, fix, or verify during a cross-topic audit.",
            MIN_SCORE,
        )

    def _answer(self, action: RAGAction) -> RAGObservation:
        content = (action.content or "").strip()
        if not content:
            return self._obs("Provide an answer grounded in the retrieved documents.", MIN_SCORE)

        self._ctx["current_answer"] = content
        matched_doc = self._find_matching_document_for_answer(content)
        self._ctx["matched_answer_doc_id"] = matched_doc["id"] if matched_doc else None
        self._ctx["answer_matches_known_doc"] = matched_doc is not None
        self._ctx["answer_is_current"] = bool(
            matched_doc and matched_doc["id"] == self._task["current_doc_id"]
        )

        if matched_doc is None:
            return self._obs(
                "That answer is not grounded in any retrieved document. Run detect to inspect conflicts.",
                0.12,
            )

        if matched_doc.get("is_outdated"):
            return self._obs(
                "That answer matches an older document revision. Run detect to explain the conflict.",
                0.2,
            )

        return self._obs(
            "That answer matches the newest policy revision. Run detect to surface any stale competing docs.",
            0.45,
        )

    def _detect(self, action: RAGAction) -> RAGObservation:
        outdated_docs = self._db.get_all_outdated(self._relevant_topics())
        if not outdated_docs:
            return self._obs("No outdated documents remain for this topic.", 0.2)

        self._ctx["detection_done"] = True
        self._ctx["hallucination_detected"] = True
        self._ctx["conflicting_doc_ids"] = [doc["id"] for doc in outdated_docs]

        reward = 0.7
        if self._ctx["matched_answer_doc_id"]:
            matched = self._db.get_document(self._ctx["matched_answer_doc_id"])
            if matched and matched.get("is_outdated"):
                reward = 0.76
        if self._mentions_conflict(action.content):
            reward += 0.03

        if self._ctx["matched_answer_doc_id"]:
            matched = self._db.get_document(self._ctx["matched_answer_doc_id"])
            if matched and matched.get("is_outdated"):
                return self._obs(
                    f"Conflict detected. Your answer matches outdated document {matched['id']}. Identify it explicitly with find_source.",
                    reward,
                )

        return self._obs(
            f"Detected {len(outdated_docs)} superseded documents for this topic. Use find_source to name the misleading one.",
            reward,
        )

    def _find(self, action: RAGAction) -> RAGObservation:
        if not self._ctx["detection_done"]:
            return self._obs("Run detect before naming the source document.", 0.1)
        if not action.target_doc_id:
            return self._obs("Specify target_doc_id for the outdated source document.", MIN_SCORE)

        doc = self._db.get_document(action.target_doc_id)
        if doc is None or doc["topic"] not in self._relevant_topics():
            return self._obs("That document is not part of the current task.", 0.12)
        if doc.get("archived", False):
            return self._obs("That document is already archived.", 0.15)
        if not doc.get("is_outdated", False):
            return self._obs("That document is current and cannot be the stale source.", 0.18)

        if self._ctx["matched_answer_doc_id"] == doc["id"]:
            self._ctx["source_doc_id"] = doc["id"]
            self._ctx["source_found"] = True
            return self._obs(
                f"Source confirmed: {doc['id']} matches the hallucinated answer. Archive it with fix.",
                0.85,
            )

        if self._ctx["matched_answer_doc_id"]:
            return self._obs(
                f"{doc['id']} is outdated, but it does not match the answer you gave. Find the exact stale source.",
                0.45,
            )

        self._ctx["source_doc_id"] = doc["id"]
        self._ctx["source_found"] = True
        return self._obs(
            f"{doc['id']} is a valid outdated source. Archive it with fix.",
            0.72,
        )

    def _fix(self, action: RAGAction) -> RAGObservation:
        if not action.target_doc_id:
            return self._obs("Specify target_doc_id before attempting a fix.", MIN_SCORE)

        result = self._db.fix_document(action.target_doc_id)
        if not result["success"]:
            return self._obs(f"Fix failed: {result['message']}", 0.15)

        if self._ctx["source_doc_id"] and action.target_doc_id != self._ctx["source_doc_id"]:
            return self._obs(
                f"Archived {action.target_doc_id}, but it was not the diagnosed source. Fix the exact source before verify.",
                0.55,
            )

        if (
            self._ctx["matched_answer_doc_id"]
            and not self._ctx["source_found"]
            and action.target_doc_id != self._ctx["matched_answer_doc_id"]
        ):
            return self._obs(
                f"Archived {action.target_doc_id}, but you have not yet identified the exact stale source.",
                0.55,
            )

        if action.target_doc_id == self._ctx["matched_answer_doc_id"]:
            self._ctx["source_found"] = True
            self._ctx["source_doc_id"] = action.target_doc_id

        self._ctx["database_fixed"] = True
        return self._obs(
            f"{result['message']}. The misleading source is archived; now verify the current answer.",
            0.9,
        )

    def _verify(self, action: RAGAction) -> RAGObservation:
        content = (action.content or "").strip()
        if not content:
            return self._obs("Provide the corrected answer during verify.", MIN_SCORE)

        if self._is_correct_answer(content):
            if not self._ctx["database_fixed"]:
                return self._obs(
                    "The answer is correct, but you still need to archive the stale source before verification can pass.",
                    0.55,
                )
            self._done = True
            self._ctx["verified_correct"] = True
            return self._obs("Pipeline complete. The answer is corrected and the stale source is archived.", MAX_SCORE)

        return self._obs("Verification failed. Ground the answer in the newest document revision.", 0.22)

    def _audit_detect(self, action: RAGAction) -> RAGObservation:
        conflicts = self._db.get_all_topics_with_conflicts(self._relevant_topics())
        total = sum(len(doc_ids) for doc_ids in conflicts.values())
        if total == 0:
            return self._obs("No outdated documents remain across the audited topics.", 0.2)

        self._ctx["detection_done"] = True
        self._ctx["hallucination_detected"] = True
        self._ctx["conflicting_doc_ids"] = [
            doc["id"] for doc in self._db.get_all_outdated(self._relevant_topics())
        ]
        topic_list = ", ".join(sorted(conflicts.keys()))
        reward = 0.72 if self._mentions_conflict(action.content) else 0.68
        return self._obs(
            f"Audit detected {total} outdated documents across: {topic_list}. Archive each stale document.",
            reward,
        )

    def _audit_find(self, action: RAGAction) -> RAGObservation:
        if not self._ctx["detection_done"]:
            return self._obs("Run detect before naming audit sources.", 0.1)
        if not action.target_doc_id:
            return self._obs("Specify target_doc_id for the outdated audit source.", MIN_SCORE)

        doc = self._db.get_document(action.target_doc_id)
        if doc is None or doc["topic"] not in self._relevant_topics():
            return self._obs("That document is outside the current audit scope.", 0.12)
        if doc.get("archived", False):
            return self._obs("That document has already been archived.", 0.15)
        if not doc.get("is_outdated", False):
            return self._obs("That document is already current.", 0.18)

        self._ctx["source_doc_id"] = doc["id"]
        self._ctx["source_found"] = True
        return self._obs(
            f"Confirmed {doc['id']} as an outdated audit target. Archive it with fix.",
            0.65,
        )

    def _audit_fix(self, action: RAGAction) -> RAGObservation:
        if not action.target_doc_id:
            return self._obs("Specify target_doc_id before fixing an audit target.", MIN_SCORE)

        result = self._db.fix_document(action.target_doc_id)
        if not result["success"]:
            return self._obs(f"Fix failed: {result['message']}", 0.12)

        self._ctx["fixes_applied"] += 1
        remaining = self._db.count_remaining_outdated(self._relevant_topics())
        total = max(1, int(self._task.get("total_outdated", 1)))
        progress = 0.5 + min(1.0, self._ctx["fixes_applied"] / total) * 0.35

        if remaining == 0:
            self._ctx["database_fixed"] = True
            return self._obs(
                "Every outdated document in the audit scope has been archived. Verify the healed KB.",
                0.9,
            )

        return self._obs(
            f"Archived {action.target_doc_id}. {remaining} outdated documents remain in the audit scope.",
            progress,
        )

    def _audit_verify(self, action: RAGAction) -> RAGObservation:
        remaining = self._db.count_remaining_outdated(self._relevant_topics())
        if remaining > 0:
            return self._obs(
                f"Audit incomplete. {remaining} outdated documents still need to be archived.",
                0.3,
            )
        if not (action.content or "").strip():
            return self._obs("Provide a short audit completion summary during verify.", MIN_SCORE)

        self._done = True
        self._ctx["database_fixed"] = True
        self._ctx["audit_completed"] = True
        self._ctx["verified_correct"] = True
        return self._obs(
            "Audit complete. All stale documents were archived and the knowledge base is consistent.",
            MAX_SCORE,
        )

    def _relevant_topics(self) -> List[str]:
        if self._task_name == "task_cross_topic_audit":
            return list(self._task.get("selected_topics", []))
        topic = self._task.get("topic")
        return [str(topic)] if topic and topic != "all" else []

    def _find_matching_document_for_answer(
        self, answer: str
    ) -> Dict[str, Any] | None:
        for doc in self._db.get_versions(self._relevant_topics()):
            answer_value = doc.get("answer_value")
            if answer_value and self._text_matches(answer, answer_value):
                return doc
        return None

    def _is_correct_answer(self, answer: str) -> bool:
        aliases = self._task.get("answer_aliases") or []
        if not aliases and self._task.get("correct_answer"):
            aliases = [self._task["correct_answer"]]
        return any(self._text_matches(answer, alias) for alias in aliases)

    @staticmethod
    def _text_matches(candidate: str, expected: str) -> bool:
        candidate_norm = RAGEnvironment._normalize_text(candidate)
        expected_norm = RAGEnvironment._normalize_text(expected)
        return bool(candidate_norm and expected_norm and expected_norm in candidate_norm)

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower().replace(",", "")
        return re.sub(r"[^a-z0-9$]+", " ", lowered).strip()

    @staticmethod
    def _mentions_conflict(text: str) -> bool:
        normalized = RAGEnvironment._normalize_text(text)
        return any(keyword in normalized for keyword in CONFLICT_KEYWORDS)

    def _clamp_reward(self, reward: float) -> float:
        return max(MIN_SCORE, min(float(reward), MAX_SCORE))

    def get_episode_score(self):
        max_steps = int(self._task["max_steps"])
        efficiency = max(0.0, 1.0 - (self._step_count / max_steps)) if max_steps else 0.0

        if self._task_name == "task_detect_hallucination":
            score = 0.05
            score += 0.20 if self._ctx["current_answer"] else 0.0
            score += 0.15 if self._ctx["answer_matches_known_doc"] else 0.0
            score += 0.30 if self._ctx["detection_done"] else 0.0
            score += 0.20 if self._ctx["hallucination_detected"] else 0.0
            score += 0.09 * efficiency
        elif self._task_name == "task_find_source":
            score = 0.05
            score += 0.15 if self._ctx["current_answer"] else 0.0
            score += 0.15 if self._ctx["answer_matches_known_doc"] else 0.0
            score += 0.20 if self._ctx["detection_done"] else 0.0
            score += 0.30 if self._ctx["source_found"] else 0.0
            score += 0.10 if self._ctx["hallucination_detected"] else 0.0
            score += 0.09 * efficiency
        elif self._task_name == "task_full_pipeline":
            score = 0.05
            score += 0.10 if self._ctx["current_answer"] else 0.0
            score += 0.15 if self._ctx["detection_done"] else 0.0
            score += 0.20 if self._ctx["source_found"] else 0.0
            score += 0.20 if self._ctx["database_fixed"] else 0.0
            score += 0.20 if self._ctx["verified_correct"] else 0.0
            score += 0.09 * efficiency
        else:
            total = max(1.0, float(self._task.get("total_outdated", 1)))
            progress = min(1.0, self._ctx["fixes_applied"] / total)
            score = 0.05
            score += 0.15 if self._ctx["detection_done"] else 0.0
            score += 0.45 * progress
            score += 0.15 if self._ctx["database_fixed"] else 0.0
            score += 0.14 if self._ctx["audit_completed"] else 0.0
            score += 0.05 * efficiency

        return round(self._clamp_reward(score), 3)

    def _obs(self, message: str, reward: float) -> RAGObservation:
        topic = self._task.get("topic", "all")
        conflicting_docs = [
            InternalDatabase._visible_document(doc)
            for doc in (
                self._db.get_document(doc_id) for doc_id in self._ctx["conflicting_doc_ids"]
            )
            if doc is not None and not doc.get("archived", False)
        ]
        return RAGObservation(
            question=self._task.get("question", ""),
            retrieved_documents=self._db.search(topic),
            current_answer=self._ctx.get("current_answer"),
            hallucination_detected=self._ctx["hallucination_detected"],
            conflicting_docs=conflicting_docs,
            database_fixed=self._ctx["database_fixed"],
            step_number=self._step_count,
            message=message,
            reward=self._clamp_reward(reward),
            done=self._done,
        )
