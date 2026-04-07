"""
server/environment.py - Self-Healing RAG Environment
4 Tasks: Easy → Medium → Hard → Expert
"""
import uuid
from openenv.core.env_server.interfaces import Environment
from models import RAGAction, RAGObservation, RAGState

ALL_TOPICS = ["leave_policy", "pricing", "refund_policy"]

class InternalDatabase:
    def __init__(self):
        self.documents = self._load_documents()
        self.fix_log = []

    def _load_documents(self):
        return {
            "hr_001": {"id":"hr_001","title":"Leave Policy v1 (2020)","content":"Employees are entitled to 10 days of annual leave per year.","date":"2020-01-01","topic":"leave_policy","is_outdated":True,"correct_doc_id":"hr_003"},
            "hr_002": {"id":"hr_002","title":"Leave Policy v2 (2022)","content":"Employees are entitled to 15 days of annual leave per year.","date":"2022-06-01","topic":"leave_policy","is_outdated":True,"correct_doc_id":"hr_003"},
            "hr_003": {"id":"hr_003","title":"Leave Policy v3 (2024) - CURRENT","content":"Employees are entitled to 20 days of annual leave per year. Effective January 2024.","date":"2024-01-01","topic":"leave_policy","is_outdated":False,"correct_doc_id":None},
            "prod_001": {"id":"prod_001","title":"Pricing 2021","content":"Our standard plan costs $50 per month.","date":"2021-03-01","topic":"pricing","is_outdated":True,"correct_doc_id":"prod_003"},
            "prod_002": {"id":"prod_002","title":"Pricing 2023","content":"Our standard plan costs $75 per month.","date":"2023-01-01","topic":"pricing","is_outdated":True,"correct_doc_id":"prod_003"},
            "prod_003": {"id":"prod_003","title":"Pricing 2024 - CURRENT","content":"Our standard plan costs $99 per month. Premium plan is $199 per month.","date":"2024-06-01","topic":"pricing","is_outdated":False,"correct_doc_id":None},
            "ref_001": {"id":"ref_001","title":"Refund Policy 2022","content":"Customers can request a refund within 7 days of purchase.","date":"2022-05-01","topic":"refund_policy","is_outdated":True,"correct_doc_id":"ref_002"},
            "ref_002": {"id":"ref_002","title":"Refund Policy 2024 - CURRENT","content":"Customers can request a refund within 30 days of purchase. No questions asked.","date":"2024-02-01","topic":"refund_policy","is_outdated":False,"correct_doc_id":None},
        }

    def search(self, topic):
        if topic == "all":
            results = [d for d in self.documents.values() if not d.get("archived", False)]
        else:
            results = [d for d in self.documents.values() if topic.lower() in d["topic"].lower() and not d.get("archived", False)]
        results.sort(key=lambda x: x["date"], reverse=True)
        return results

    def get_all_versions(self, topic):
        return [d for d in self.documents.values() if d["topic"] == topic]

    def get_all_outdated(self):
        """Get ALL outdated docs across ALL topics — used by Task 4."""
        return [d for d in self.documents.values() if d.get("is_outdated") and not d.get("archived", False)]

    def get_all_topics_with_conflicts(self):
        """Returns dict of topic -> list of outdated doc ids."""
        conflicts = {}
        for topic in ALL_TOPICS:
            outdated = [d["id"] for d in self.get_all_versions(topic) if d.get("is_outdated") and not d.get("archived", False)]
            if outdated:
                conflicts[topic] = outdated
        return conflicts

    def fix_document(self, wrong_doc_id):
        if wrong_doc_id not in self.documents:
            return {"success": False, "message": "Document not found"}
        doc = self.documents[wrong_doc_id]
        correct_id = doc.get("correct_doc_id")
        if not correct_id:
            return {"success": False, "message": "No newer version found"}
        self.documents[wrong_doc_id]["archived"] = True
        self.fix_log.append({"archived": wrong_doc_id, "promoted": correct_id})
        return {"success": True, "message": f"Archived {wrong_doc_id}, promoted {correct_id}", "correct_doc": self.documents[correct_id]}

    def count_remaining_outdated(self):
        """How many outdated docs still not fixed."""
        return len([d for d in self.documents.values() if d.get("is_outdated") and not d.get("archived", False)])


TASKS = {
    # ── TASK 1: EASY ──────────────────────────────────────────
    "task_detect_hallucination": {
        "question": "How many days of annual leave do employees get?",
        "topic": "leave_policy",
        "correct_answer": "20 days",
        "difficulty": "easy",
        "max_steps": 4,
        "passing_score": 0.6,
        "description": "Detect if the AI answer contains hallucination from outdated internal documents"
    },
    # ── TASK 2: MEDIUM ────────────────────────────────────────
    "task_find_source": {
        "question": "What is the current price of the standard plan?",
        "topic": "pricing",
        "correct_answer": "$99 per month",
        "wrong_doc_id": "prod_001",
        "difficulty": "medium",
        "max_steps": 6,
        "passing_score": 0.7,
        "description": "Find which specific internal document is causing the hallucination"
    },
    # ── TASK 3: HARD ──────────────────────────────────────────
    "task_full_pipeline": {
        "question": "What is our refund policy?",
        "topic": "refund_policy",
        "correct_answer": "30 days",
        "wrong_doc_id": "ref_001",
        "difficulty": "hard",
        "max_steps": 10,
        "passing_score": 0.85,
        "description": "Full self-healing pipeline: detect, find source, fix database, verify"
    },
    # ── TASK 4: EXPERT (NEW!) ─────────────────────────────────
    "task_cross_topic_audit": {
        "question": "Perform a full audit of the entire knowledge base. Find and fix ALL outdated documents across ALL topics.",
        "topic": "all",
        "correct_answer": "audit_complete",
        "difficulty": "expert",
        "max_steps": 15,
        "passing_score": 0.90,
        "description": "Autonomous full database audit: find and fix ALL outdated docs across ALL topics without being told which topic to check",
        # Expected: agent must fix hr_001, hr_002, prod_001, prod_002, ref_001 (5 outdated docs)
        "total_outdated": 5,
    },
}


class RAGEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self._db = InternalDatabase()
        self._task = TASKS["task_detect_hallucination"]
        self._ctx = {
            "hallucination_detected": False,
            "database_fixed": False,
            "current_answer": None,
            "conflicting_docs": [],
            # Task 4 specific
            "fixes_applied": 0,
            "topics_audited": [],
            "audit_complete": False,
        }
        self._step_count = 0
        self._episode_id = ""
        self._rewards = []
        self._done = False

    def reset(self, **kwargs) -> RAGObservation:
        task_name = kwargs.get("task_name", "task_detect_hallucination")
        self._db = InternalDatabase()
        self._task = TASKS.get(task_name, TASKS["task_detect_hallucination"])
        self._ctx = {
            "hallucination_detected": False,
            "database_fixed": False,
            "current_answer": None,
            "conflicting_docs": [],
            "fixes_applied": 0,
            "topics_audited": [],
            "audit_complete": False,
        }
        self._step_count = 0
        self._episode_id = str(uuid.uuid4())
        self._rewards = []
        self._done = False

        docs = self._db.search(self._task["topic"])

        if task_name == "task_cross_topic_audit":
            message = (
                "EXPERT AUDIT TASK: Scan ALL topics in the knowledge base. "
                "Find every outdated document and fix them all. "
                "Topics to audit: leave_policy, pricing, refund_policy. "
                "Use detect to find conflicts, find_source to identify docs, "
                "fix to archive them, verify when all are fixed."
            )
        else:
            message = "Episode started. Analyze the documents and answer the question."

        return RAGObservation(
            question=self._task["question"],
            retrieved_documents=docs,
            step_number=0,
            message=message,
            reward=0.05,
        )

    def step(self, action: RAGAction) -> RAGObservation:
        self._step_count += 1

        # Route to task 4 handlers if expert task
        if self._task.get("difficulty") == "expert":
            if action.action_type == "detect":
                obs = self._audit_detect(action)
            elif action.action_type == "find_source":
                obs = self._audit_find(action)
            elif action.action_type == "fix":
                obs = self._audit_fix(action)
            elif action.action_type == "verify":
                obs = self._audit_verify(action)
            else:
                obs = self._make_obs("Use detect, find_source, fix, or verify actions.", 0.05)
        else:
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
                obs = self._make_obs("Unknown action type.", 0.05)

        self._rewards.append(obs.reward)
        if self._step_count >= self._task["max_steps"]:
            self._done = True
        return obs

    @property
    def state(self):
        return RAGState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            current_task=self._task.get("difficulty", ""),
            hallucination_detected=self._ctx["hallucination_detected"],
            database_fixed=self._ctx["database_fixed"],
            fix_log=self._db.fix_log,
            episode_rewards=self._rewards,
            done=self._done,
        )

    # ── Standard Task Handlers (Tasks 1-3) ────────────────────

    def _handle_answer(self, action):
        self._ctx["current_answer"] = action.content
        correct = self._task["correct_answer"].lower()
        given = action.content.lower()
        if correct in given or given in correct:
            return self._make_obs("Answer correct! Now check for hallucination risk.", 0.6)
        self._ctx["hallucination_detected"] = True
        return self._make_obs("Wrong answer — likely from outdated document. Run detect.", 0.1)

    def _handle_detect(self, action):
        outdated = [d for d in self._db.get_all_versions(self._task["topic"]) if d.get("is_outdated")]
        if outdated:
            self._ctx["conflicting_docs"] = outdated
            self._ctx["hallucination_detected"] = True
            return self._make_obs(f"Detected {len(outdated)} outdated documents. Find the source.", 0.7)
        return self._make_obs("No conflicts found.", 0.2)

    def _handle_find_source(self, action):
        outdated_ids = [d["id"] for d in self._db.get_all_versions(self._task["topic"]) if d.get("is_outdated")]
        known_wrong = self._task.get("wrong_doc_id", "")
        if action.target_doc_id == known_wrong:
            return self._make_obs(f"Correct source found: {action.target_doc_id}. Now fix the database.", 0.8)
        elif action.target_doc_id in outdated_ids:
            return self._make_obs("Partially correct — outdated but not primary source.", 0.4)
        return self._make_obs("Wrong document. Try again.", 0.2)

    def _handle_fix(self, action):
        if not action.target_doc_id:
            return self._make_obs("Specify target_doc_id to fix.", 0.05)
        result = self._db.fix_document(action.target_doc_id)
        if result["success"]:
            self._ctx["database_fixed"] = True
            return self._make_obs(f"Database fixed! {result['message']}. Now verify.", 0.9)
        return self._make_obs(f"Fix failed: {result['message']}", 0.2)

    def _handle_verify(self, action):
        if not self._ctx["database_fixed"]:
            return self._make_obs("Fix the database before verifying.", 0.05)
        correct = self._task["correct_answer"].lower()
        given = action.content.lower()
        if correct in given or given in correct:
            self._done = True
            return self._make_obs("COMPLETE! Full self-healing pipeline succeeded!", 0.99)
        return self._make_obs("Fix applied but answer still incorrect.", 0.5)

    # ── Task 4: Expert Audit Handlers ─────────────────────────

    def _audit_detect(self, action):
        """Agent scans all topics for conflicts."""
        conflicts = self._db.get_all_topics_with_conflicts()
        total_outdated = sum(len(v) for v in conflicts.values())

        if total_outdated > 0:
            self._ctx["hallucination_detected"] = True
            self._ctx["conflicting_docs"] = self._db.get_all_outdated()
            topics_found = list(conflicts.keys())
            self._ctx["topics_audited"] = topics_found
            return self._make_obs(
                f"AUDIT FOUND: {total_outdated} outdated documents across {len(topics_found)} topics: {topics_found}. "
                f"Fix each one using find_source + fix actions.",
                0.5
            )
        return self._make_obs("No conflicts found across any topic.", 0.1)

    def _audit_find(self, action):
        """Agent identifies a specific outdated document."""
        all_outdated = self._db.get_all_outdated()
        outdated_ids = [d["id"] for d in all_outdated]

        if action.target_doc_id in outdated_ids:
            remaining = len(outdated_ids)
            return self._make_obs(
                f"Confirmed: {action.target_doc_id} is outdated. Fix it now. "
                f"Total remaining outdated docs: {remaining}",
                0.6
            )
        return self._make_obs(f"Document {action.target_doc_id} is not outdated or already fixed.", 0.2)

    def _audit_fix(self, action):
        """Agent fixes one outdated document at a time."""
        if not action.target_doc_id:
            return self._make_obs("Specify target_doc_id to fix.", 0.05)

        result = self._db.fix_document(action.target_doc_id)
        if result["success"]:
            self._ctx["fixes_applied"] += 1
            remaining = self._db.count_remaining_outdated()

            # Partial reward based on progress
            total = self._task.get("total_outdated", 5)
            fixed = self._ctx["fixes_applied"]
            progress_reward = 0.5 + (fixed / total) * 0.4  # 0.5 to 0.9

            if remaining == 0:
                self._ctx["database_fixed"] = True
                return self._make_obs(
                    f"Fixed {action.target_doc_id}! ALL {total} outdated documents fixed! Now verify.",
                    0.9
                )
            return self._make_obs(
                f"Fixed {action.target_doc_id}! {remaining} outdated documents still remain. Keep fixing.",
                round(progress_reward, 2)
            )
        return self._make_obs(f"Fix failed: {result['message']}", 0.1)

    def _audit_verify(self, action):
        """Agent verifies the entire database is clean."""
        remaining = self._db.count_remaining_outdated()

        if remaining == 0:
            self._ctx["audit_complete"] = True
            self._done = True
            fixes = self._ctx["fixes_applied"]
            return self._make_obs(
                f"EXPERT AUDIT COMPLETE! Fixed {fixes} outdated documents across all topics. "
                f"Knowledge base is fully clean and self-healed!",
                0.99
            )
        return self._make_obs(
            f"Audit not complete. {remaining} outdated documents still exist. Fix them first.",
            0.3
        )

    # ── Helper ────────────────────────────────────────────────

    def _make_obs(self, message, reward):
        topic = self._task.get("topic", "all")
        docs = self._db.search(topic)
        return RAGObservation(
            question=self._task.get("question", ""),
            retrieved_documents=docs,
            current_answer=self._ctx.get("current_answer"),
            hallucination_detected=self._ctx["hallucination_detected"],
            conflicting_docs=self._ctx["conflicting_docs"],
            database_fixed=self._ctx["database_fixed"],
            step_number=self._step_count,
            message=message,
            reward=reward,
            done=self._done,
        )