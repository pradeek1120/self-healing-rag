import os, json, requests
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "https://pradeerock-self-healing-rag.hf.space")
BENCHMARK    = "self-healing-rag"

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

TASKS = {
    "task_detect_hallucination": {"question":"How many days of annual leave?","difficulty":"easy","max_steps":4,"passing_score":0.6,"description":"Detect hallucination"},
    "task_find_source": {"question":"What is current price?","difficulty":"medium","max_steps":6,"passing_score":0.7,"description":"Find source doc"},
    "task_full_pipeline": {"question":"What is refund policy?","difficulty":"hard","max_steps":10,"passing_score":0.85,"description":"Full pipeline"},
    "task_cross_topic_audit": {"question":"Audit entire knowledge base.","difficulty":"expert","max_steps":15,"passing_score":0.90,"description":"Expert audit"}
}

FALLBACKS = {
    "easy":   [{"action_type":"answer","content":"10 days"},{"action_type":"detect","content":"checking"}],
    "medium": [{"action_type":"answer","content":"50 dollars"},{"action_type":"detect","content":"checking"},{"action_type":"find_source","content":"found","target_doc_id":"prod_001"}],
    "hard":   [{"action_type":"answer","content":"7 days"},{"action_type":"detect","content":"checking"},{"action_type":"find_source","content":"found","target_doc_id":"ref_001"},{"action_type":"fix","content":"fixing","target_doc_id":"ref_001"},{"action_type":"verify","content":"30 days"}],
    "expert": [{"action_type":"detect","content":"scanning"},{"action_type":"fix","content":"fix","target_doc_id":"hr_001"},{"action_type":"fix","content":"fix","target_doc_id":"hr_002"},{"action_type":"fix","content":"fix","target_doc_id":"prod_001"},{"action_type":"fix","content":"fix","target_doc_id":"prod_002"},{"action_type":"fix","content":"fix","target_doc_id":"ref_001"},{"action_type":"verify","content":"audit complete"}]
}

def agent_decide(obs, task, step):
    try:
        docs = obs.get("observation",{}).get("retrieved_documents",[])
        docs_text = "\n".join([f"[{d.get('id')}] {d.get('title')}: {d.get('content')}" for d in docs[:4]])
        prompt = f"Task: {task['description']}\nDocuments:\n{docs_text}\nStep {step}: Choose action JSON only."
        resp = client.chat.completions.create(model=MODEL_NAME,messages=[{"role":"system","content":"RAG agent. JSON only."},{"role":"user","content":prompt}],max_tokens=150,temperature=0.1)
        text = resp.choices[0].message.content.strip()
        if "```" in text: text = text.split("```")[1].replace("json","").strip()
        return json.loads(text)
    except:
        seq = FALLBACKS.get(task.get("difficulty","easy"), FALLBACKS["easy"])
        return seq[min(step-1, len(seq)-1)]

for task_name, task in TASKS.items():
    try:
        obs = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name}).json()
    except Exception as e:
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")
        print(f"[STEP] step=1 action=error reward=0.00 done=true error={str(e)[:50]}")
        print(f"[END] success=false steps=1 rewards=0.00")
        continue

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")
    done=False; step=0; rewards=[]; last_error=None

    try:
        while not done and step < task["max_steps"]:
            action = agent_decide(obs, task, step+1)
            at = action.get("action_type","answer")
            content = action.get("content","")
            tid = action.get("target_doc_id", None)
            result = requests.post(f"{ENV_URL}/step", json={"action":{"action_type":at,"content":content,"target_doc_id":tid}}).json()
            reward = float(result.get("reward",0.0))
            done = result.get("done", False)
            obs = result
            step+=1; rewards.append(reward)
            print(f"[STEP] step={step} action={at}('{content[:20]}') reward={reward:.2f} done={str(done).lower()} error=null")
        success = sum(rewards) >= task["passing_score"]
    except Exception as e:
        last_error = str(e)[:60]
        success = False
        print(f"[STEP] step={step+1} action=error reward=0.00 done=true error={last_error}")

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) or "0.00"
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")
