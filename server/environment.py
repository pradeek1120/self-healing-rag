import uuid
from openenv.core.env_server.interfaces import Environment

try:
    from models import RAGAction, RAGObservation, RAGState
    from tasks import TASKS
except ImportError:
    from ..models import RAGAction, RAGObservation, RAGState
    from ..tasks import TASKS

ALL_TOPICS = ["leave_policy", "pricing", "refund_policy"]
MIN_SCORE = 0.01
MAX_SCORE = 0.99

class InternalDatabase:
    def __init__(self):
        self.documents = {
            "hr_001":{"id":"hr_001","title":"Leave Policy v1 (2020)","content":"Employees are entitled to 10 days of annual leave per year.","date":"2020-01-01","topic":"leave_policy","is_outdated":True,"correct_doc_id":"hr_003"},
            "hr_002":{"id":"hr_002","title":"Leave Policy v2 (2022)","content":"Employees are entitled to 15 days of annual leave per year.","date":"2022-06-01","topic":"leave_policy","is_outdated":True,"correct_doc_id":"hr_003"},
            "hr_003":{"id":"hr_003","title":"Leave Policy v3 (2024) - CURRENT","content":"Employees are entitled to 20 days of annual leave per year. Effective January 2024.","date":"2024-01-01","topic":"leave_policy","is_outdated":False,"correct_doc_id":None},
            "prod_001":{"id":"prod_001","title":"Pricing 2021","content":"Our standard plan costs $50 per month.","date":"2021-03-01","topic":"pricing","is_outdated":True,"correct_doc_id":"prod_003"},
            "prod_002":{"id":"prod_002","title":"Pricing 2023","content":"Our standard plan costs $75 per month.","date":"2023-01-01","topic":"pricing","is_outdated":True,"correct_doc_id":"prod_003"},
            "prod_003":{"id":"prod_003","title":"Pricing 2024 - CURRENT","content":"Our standard plan costs $99 per month.","date":"2024-06-01","topic":"pricing","is_outdated":False,"correct_doc_id":None},
            "ref_001":{"id":"ref_001","title":"Refund Policy 2022","content":"Customers can request a refund within 7 days of purchase.","date":"2022-05-01","topic":"refund_policy","is_outdated":True,"correct_doc_id":"ref_002"},
            "ref_002":{"id":"ref_002","title":"Refund Policy 2024 - CURRENT","content":"Customers can request a refund within 30 days of purchase. No questions asked.","date":"2024-02-01","topic":"refund_policy","is_outdated":False,"correct_doc_id":None},
        }
        self.fix_log = []

    def search(self, topic):
        if topic == "all":
            r = [d for d in self.documents.values() if not d.get("archived", False)]
        else:
            r = [d for d in self.documents.values() if topic in d["topic"] and not d.get("archived", False)]
        return sorted(r, key=lambda x: x["date"], reverse=True)

    def get_all_versions(self, topic):
        return [d for d in self.documents.values() if d["topic"] == topic]

    def get_all_outdated(self):
        return [d for d in self.documents.values() if d.get("is_outdated") and not d.get("archived", False)]

    def get_all_topics_with_conflicts(self):
        c = {}
        for t in ALL_TOPICS:
            o = [d["id"] for d in self.get_all_versions(t) if d.get("is_outdated") and not d.get("archived", False)]
            if o:
                c[t] = o
        return c

    def fix_document(self, doc_id):
        if doc_id not in self.documents:
            return {"success": False, "message": "Not found"}
        doc = self.documents[doc_id]
        cid = doc.get("correct_doc_id")
        if not cid:
            return {"success": False, "message": "No newer version"}
        self.documents[doc_id]["archived"] = True
        self.fix_log.append({"archived": doc_id, "promoted": cid})
        return {"success": True, "message": f"Archived {doc_id}, promoted {cid}", "correct_doc": self.documents[cid]}

    def count_remaining_outdated(self):
        return len([d for d in self.documents.values() if d.get("is_outdated") and not d.get("archived", False)])

class RAGEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._db = InternalDatabase()
        self._task_name = "task_detect_hallucination"
        self._task = TASKS["task_detect_hallucination"]
        self._ctx = self._fresh_context()
        self._step_count = 0
        self._episode_id = ""
        self._rewards = []
        self._done = False
        self._actions = []

    def _fresh_context(self):
        return {
            "hallucination_detected": False,
            "database_fixed": False,
            "current_answer": None,
            "conflicting_docs": [],
            "fixes_applied": 0,
            "source_found": False,
            "verified_correct": False,
            "audit_completed": False,
        }

    def reset(self, **kwargs):
        task_name = kwargs.get("task_name", "task_detect_hallucination")
        self._db = InternalDatabase()
        self._task_name = task_name if task_name in TASKS else "task_detect_hallucination"
        self._task = TASKS.get(task_name, TASKS["task_detect_hallucination"])
        self._ctx = self._fresh_context()
        self._step_count = 0
        self._episode_id = str(uuid.uuid4())
        self._rewards = []
        self._done = False
        self._actions = []
        docs = self._db.search(self._task["topic"])
        msg = "EXPERT AUDIT: Scan ALL topics, find and fix ALL outdated docs." if task_name == "task_cross_topic_audit" else "Episode started. Analyze documents and answer the question."
        return RAGObservation(
            question=self._task["question"],
            retrieved_documents=docs,
            step_number=0,
            message=msg,
            reward=self._clamp_reward(MIN_SCORE),
            done=False,
        )

    def step(self, action):
        self._step_count += 1
        self._actions.append(
            {
                "action_type": action.action_type,
                "content": action.content,
                "target_doc_id": action.target_doc_id,
            }
        )
        expert = self._task.get("difficulty") == "expert"
        if expert:
            if action.action_type == "detect": obs = self._audit_detect(action)
            elif action.action_type == "find_source": obs = self._audit_find(action)
            elif action.action_type == "fix": obs = self._audit_fix(action)
            elif action.action_type == "verify": obs = self._audit_verify(action)
            else: obs = self._obs("Use detect, find_source, fix, or verify.", MIN_SCORE)
        else:
            if action.action_type == "answer": obs = self._answer(action)
            elif action.action_type == "detect": obs = self._detect(action)
            elif action.action_type == "find_source": obs = self._find(action)
            elif action.action_type == "fix": obs = self._fix(action)
            elif action.action_type == "verify": obs = self._verify(action)
            else: obs = self._obs("Unknown action.", MIN_SCORE)
        self._rewards.append(obs.reward)
        if self._step_count >= self._task["max_steps"]: self._done = True
        return obs

    @property
    def state(self):
        return RAGState(episode_id=self._episode_id, step_count=self._step_count, current_task=self._task_name, hallucination_detected=self._ctx["hallucination_detected"], database_fixed=self._ctx["database_fixed"], fix_log=self._db.fix_log, episode_rewards=self._rewards, done=self._done)

    def _answer(self, a):
        self._ctx["current_answer"] = a.content
        if self._task["correct_answer"].lower() in a.content.lower():
            return self._obs("Correct! Check for hallucination risk.", 0.6)
        self._ctx["hallucination_detected"] = True
        return self._obs("Wrong answer — likely from outdated doc. Run detect.", 0.1)

    def _detect(self, a):
        outdated = [d for d in self._db.get_all_versions(self._task["topic"]) if d.get("is_outdated")]
        if outdated:
            self._ctx["conflicting_docs"] = outdated
            self._ctx["hallucination_detected"] = True
            return self._obs(f"Detected {len(outdated)} outdated docs. Find source.", 0.7)
        return self._obs("No conflicts found.", 0.2)

    def _find(self, a):
        outdated_ids = [d["id"] for d in self._db.get_all_versions(self._task["topic"]) if d.get("is_outdated")]
        wrong = self._task.get("wrong_doc_id", "")
        if a.target_doc_id == wrong:
            self._ctx["source_found"] = True
            return self._obs(f"Source found: {a.target_doc_id}. Fix now.", 0.8)
        if a.target_doc_id in outdated_ids:
            return self._obs("Partially correct.", 0.4)
        return self._obs("Wrong document.", 0.2)

    def _fix(self, a):
        if not a.target_doc_id:
            return self._obs("Specify target_doc_id.", MIN_SCORE)
        r = self._db.fix_document(a.target_doc_id)
        if r["success"]:
            self._ctx["database_fixed"] = True
            return self._obs(f"Fixed! {r['message']}. Verify now.", 0.9)
        return self._obs(f"Fix failed: {r['message']}", 0.2)

    def _verify(self, a):
        if self._task["correct_answer"].lower() in a.content.lower():
            self._done = True
            self._ctx["verified_correct"] = True
            return self._obs("COMPLETE! Pipeline succeeded!", MAX_SCORE)
        return self._obs("Answer incorrect. Check latest documents.", 0.5)

    def _audit_detect(self, a):
        conflicts = self._db.get_all_topics_with_conflicts()
        total = sum(len(v) for v in conflicts.values())
        if total > 0:
            self._ctx["hallucination_detected"] = True
            self._ctx["conflicting_docs"] = self._db.get_all_outdated()
            return self._obs(f"AUDIT: {total} outdated docs found across {list(conflicts.keys())}. Fix each one.", 0.7)
        return self._obs("No conflicts found.", 0.1)

    def _audit_find(self, a):
        outdated_ids = [d["id"] for d in self._db.get_all_outdated()]
        if a.target_doc_id in outdated_ids:
            self._ctx["source_found"] = True
            return self._obs(f"Confirmed {a.target_doc_id} is outdated. Fix it.", 0.6)
        return self._obs(f"{a.target_doc_id} not outdated or already fixed.", 0.2)

    def _audit_fix(self, a):
        if not a.target_doc_id:
            return self._obs("Specify target_doc_id.", MIN_SCORE)
        r = self._db.fix_document(a.target_doc_id)
        if r["success"]:
            self._ctx["fixes_applied"] += 1
            remaining = self._db.count_remaining_outdated()
            total = self._task.get("total_outdated", 5)
            fixed = self._ctx["fixes_applied"]
            progress = round(0.5 + (fixed / total) * 0.4, 2)
            if remaining == 0:
                self._ctx["database_fixed"] = True
                return self._obs(f"ALL {total} docs fixed! Verify now.", 0.9)
            return self._obs(f"Fixed {a.target_doc_id}! {remaining} remaining.", progress)
        return self._obs(f"Fix failed: {r['message']}", 0.1)

    def _audit_verify(self, a):
        remaining = self._db.count_remaining_outdated()
        if remaining == 0:
            self._done = True
            self._ctx["audit_completed"] = True
            self._ctx["verified_correct"] = True
            return self._obs("EXPERT AUDIT COMPLETE! Knowledge base fully healed!", MAX_SCORE)
        return self._obs(f"Audit incomplete. {remaining} outdated docs remain.", 0.3)

    def _clamp_reward(self, reward):
        return max(MIN_SCORE, min(float(reward), MAX_SCORE))

    def get_episode_score(self):
        max_steps = int(self._task["max_steps"])
        efficiency = max(0.0, 1.0 - (self._step_count / max_steps)) if max_steps else 0.0

        if self._task_name == "task_detect_hallucination":
            score = 0.05
            score += 0.10 if any(a["action_type"] == "answer" for a in self._actions) else 0.0
            score += 0.25 if self._ctx["hallucination_detected"] else 0.0
            score += 0.25 if self._ctx["conflicting_docs"] else 0.0
            score += 0.20 if any(a["action_type"] == "detect" for a in self._actions) else 0.0
            score += 0.14 * efficiency
        elif self._task_name == "task_find_source":
            score = 0.05
            score += 0.15 if self._ctx["hallucination_detected"] else 0.0
            score += 0.15 if any(a["action_type"] == "detect" for a in self._actions) else 0.0
            score += 0.20 if any(a["action_type"] == "find_source" for a in self._actions) else 0.0
            score += 0.35 if self._ctx["source_found"] else 0.0
            score += 0.09 * efficiency
        elif self._task_name == "task_full_pipeline":
            score = 0.05
            score += 0.10 if self._ctx["hallucination_detected"] else 0.0
            score += 0.20 if self._ctx["source_found"] else 0.0
            score += 0.25 if self._ctx["database_fixed"] else 0.0
            score += 0.25 if self._ctx["verified_correct"] else 0.0
            score += 0.14 * efficiency
        else:
            total = float(self._task.get("total_outdated", 5))
            progress = min(1.0, self._ctx["fixes_applied"] / total) if total else 0.0
            score = 0.05
            score += 0.10 if self._ctx["hallucination_detected"] else 0.0
            score += 0.50 * progress
            score += 0.15 if self._ctx["database_fixed"] else 0.0
            score += 0.14 if self._ctx["audit_completed"] else 0.0
            score += 0.05 * efficiency

        return round(self._clamp_reward(score), 3)

    def _obs(self, message, reward):
        docs = self._db.search(self._task.get("topic", "all"))
        return RAGObservation(
            question=self._task.get("question", ""),
            retrieved_documents=docs,
            current_answer=self._ctx.get("current_answer"),
            hallucination_detected=self._ctx["hallucination_detected"],
            conflicting_docs=self._ctx["conflicting_docs"],
            database_fixed=self._ctx["database_fixed"],
            step_number=self._step_count,
            message=message,
            reward=self._clamp_reward(reward),
            done=self._done,
        )
