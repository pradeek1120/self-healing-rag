import uuid
from openenv.core.env_server.interfaces import Environment
from models import RAGAction, RAGObservation, RAGState

class InternalDatabase:
    def __init__(self):
        self.documents = {"hr_001":{"id":"hr_001","title":"Leave Policy v1 (2020)","content":"Employees are entitled to 10 days of annual leave per year.","date":"2020-01-01","topic":"leave_policy","is_outdated":True,"correct_doc_id":"hr_003"},"hr_003":{"id":"hr_003","title":"Leave Policy v3 (2024) - CURRENT","content":"Employees are entitled to 20 days of annual leave per year. Effective January 2024.","date":"2024-01-01","topic":"leave_policy","is_outdated":False,"correct_doc_id":None},"prod_001":{"id":"prod_001","title":"Pricing 2021","content":"Our standard plan costs $50 per month.","date":"2021-03-01","topic":"pricing","is_outdated":True,"correct_doc_id":"prod_003"},"prod_003":{"id":"prod_003","title":"Pricing 2024 - CURRENT","content":"Our standard plan costs $99 per month.","date":"2024-06-01","topic":"pricing","is_outdated":False,"correct_doc_id":None},"ref_001":{"id":"ref_001","title":"Refund Policy 2022","content":"Customers can request a refund within 7 days of purchase.","date":"2022-05-01","topic":"refund_policy","is_outdated":True,"correct_doc_id":"ref_002"},"ref_002":{"id":"ref_002","title":"Refund Policy 2024 - CURRENT","content":"Customers can request a refund within 30 days of purchase.","date":"2024-02-01","topic":"refund_policy","is_outdated":False,"correct_doc_id":None}}
        self.fix_log = []
    def search(self,topic):
        r=[d for d in self.documents.values() if topic in d["topic"] and not d.get("archived",False)]
        return sorted(r,key=lambda x:x["date"],reverse=True)
    def get_all_versions(self,topic):
        return [d for d in self.documents.values() if d["topic"]==topic]
    def fix_document(self,doc_id):
        if doc_id not in self.documents: return {"success":False,"message":"Not found"}
        doc=self.documents[doc_id]; cid=doc.get("correct_doc_id")
        if not cid: return {"success":False,"message":"No newer version"}
        self.documents[doc_id]["archived"]=True
        self.fix_log.append({"archived":doc_id,"promoted":cid})
        return {"success":True,"message":f"Fixed {doc_id}","correct_doc":self.documents[cid]}

TASKS={"task_detect_hallucination":{"question":"How many days of annual leave do employees get?","topic":"leave_policy","correct_answer":"20 days","difficulty":"easy","max_steps":4,"passing_score":0.6},"task_find_source":{"question":"What is the current price of the standard plan?","topic":"pricing","correct_answer":"$99 per month","wrong_doc_id":"prod_001","difficulty":"medium","max_steps":6,"passing_score":0.7},"task_full_pipeline":{"question":"What is our refund policy?","topic":"refund_policy","correct_answer":"30 days","wrong_doc_id":"ref_001","difficulty":"hard","max_steps":10,"passing_score":0.85}}

class RAGEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._db=InternalDatabase(); self._task=TASKS["task_detect_hallucination"]
        self._ctx={"hallucination_detected":False,"database_fixed":False,"current_answer":None,"conflicting_docs":[]}
        self._step_count=0; self._episode_id=""; self._rewards=[]; self._done=False

    def reset(self,**kwargs):
        task_name=kwargs.get("task_name","task_detect_hallucination")
        self._db=InternalDatabase(); self._task=TASKS.get(task_name,TASKS["task_detect_hallucination"])
        self._ctx={"hallucination_detected":False,"database_fixed":False,"current_answer":None,"conflicting_docs":[]}
        self._step_count=0; self._episode_id=str(uuid.uuid4()); self._rewards=[]; self._done=False
        docs=self._db.search(self._task["topic"])
        return RAGObservation(question=self._task["question"],retrieved_documents=docs,step_number=0,message="Episode started.",reward=0.0)

    def step(self,action):
        self._step_count+=1
        if action.action_type=="answer": obs=self._answer(action)
        elif action.action_type=="detect": obs=self._detect(action)
        elif action.action_type=="find_source": obs=self._find(action)
        elif action.action_type=="fix": obs=self._fix(action)
        elif action.action_type=="verify": obs=self._verify(action)
        else: obs=self._obs("Unknown action",0.0)
        self._rewards.append(obs.reward)
        if self._step_count>=self._task["max_steps"]: self._done=True
        return obs

    @property
    def state(self):
        return RAGState(episode_id=self._episode_id,step_count=self._step_count,current_task=self._task.get("difficulty",""),hallucination_detected=self._ctx["hallucination_detected"],database_fixed=self._ctx["database_fixed"],fix_log=self._db.fix_log,episode_rewards=self._rewards,done=self._done)

    def _answer(self,a):
        self._ctx["current_answer"]=a.content
        if self._task["correct_answer"].lower() in a.content.lower(): return self._obs("Correct! Check for hallucination risk.",0.6)
        self._ctx["hallucination_detected"]=True; return self._obs("Wrong answer - likely from outdated doc. Run detect.",0.1)

    def _detect(self,a):
        outdated=[d for d in self._db.get_all_versions(self._task["topic"]) if d.get("is_outdated")]
        if outdated:
            self._ctx["conflicting_docs"]=outdated; self._ctx["hallucination_detected"]=True
            return self._obs(f"Detected {len(outdated)} outdated docs. Find source.",0.7)
        return self._obs("No conflicts found.",0.2)

    def _find(self,a):
        wrong=self._task.get("wrong_doc_id","")
        outdated_ids=[d["id"] for d in self._db.get_all_versions(self._task["topic"]) if d.get("is_outdated")]
        if a.target_doc_id==wrong: return self._obs(f"Source found: {a.target_doc_id}. Now fix.",0.8)
        if a.target_doc_id in outdated_ids: return self._obs("Partially correct.",0.4)
        return self._obs("Wrong document.",0.2)

    def _fix(self,a):
        if not a.target_doc_id: return self._obs("Specify target_doc_id.",0.0)
        r=self._db.fix_document(a.target_doc_id)
        if r["success"]: self._ctx["database_fixed"]=True; return self._obs(f"Fixed! {r['message']}. Verify now.",0.9)
        return self._obs(f"Fix failed: {r['message']}",0.2)

    def _verify(self,a):
        if not self._ctx["database_fixed"]: return self._obs("Fix database first.",0.0)
        if self._task["correct_answer"].lower() in a.content.lower(): self._done=True; return self._obs("COMPLETE! Pipeline succeeded!",1.0)
        return self._obs("Still incorrect.",0.5)

    def _obs(self,msg,reward):
        docs=self._db.search(self._task["topic"])
        return RAGObservation(question=self._task.get("question",""),retrieved_documents=docs,current_answer=self._ctx.get("current_answer"),hallucination_detected=self._ctx["hallucination_detected"],conflicting_docs=self._ctx["conflicting_docs"],database_fixed=self._ctx["database_fixed"],step_number=self._step_count,message=msg,reward=reward)
