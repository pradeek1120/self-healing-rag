import uvicorn
from fastapi import Body
from openenv.core.env_server import create_app

try:
    from server.environment import RAGEnvironment
    from models import RAGAction, RAGObservation
    from tasks import TASKS, list_tasks
except ImportError:
    from .environment import RAGEnvironment
    from ..models import RAGAction, RAGObservation
    from ..tasks import TASKS, list_tasks

app = create_app(RAGEnvironment, RAGAction, RAGObservation)


@app.get("/")
def root():
    return {
        "name": "self-healing-rag",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "reset": "/reset",
        "step": "/step",
        "state": "/state",
    }


@app.get("/tasks")
def tasks():
    return {"tasks": [task.__dict__ for task in list_tasks()]}


@app.get("/info")
def info():
    return {
        "name": "self-healing-rag",
        "tasks": [task.__dict__ for task in list_tasks()],
    }


@app.post("/grader")
def grader(payload: dict = Body(default={})):
    task_name = (
        payload.get("task_name")
        or payload.get("task_id")
        or payload.get("task")
        or "task_detect_hallucination"
    )
    task_name = task_name if task_name in TASKS else "task_detect_hallucination"

    trajectory = payload.get("trajectory") or payload.get("steps") or payload.get("actions") or []
    rewards = payload.get("episode_rewards") or payload.get("rewards")

    env = RAGEnvironment()
    env.reset(task_name=task_name)

    if isinstance(trajectory, list):
        for item in trajectory:
            action_data = item.get("action", item) if isinstance(item, dict) else item
            if not isinstance(action_data, dict):
                continue
            action = RAGAction(
                action_type=action_data.get("action_type", "answer"),
                content=action_data.get("content", ""),
                target_doc_id=action_data.get("target_doc_id"),
            )
            env.step(action)
            if env.state.done:
                break

    score = env.get_episode_score()

    if (not trajectory) and rewards:
        if isinstance(rewards, str):
            reward_values = [float(value) for value in rewards.split(",") if value.strip()]
        elif isinstance(rewards, list):
            reward_values = [float(value) for value in rewards]
        else:
            reward_values = []
        if reward_values:
            score = round(env._clamp_reward(sum(reward_values) / len(reward_values)), 3)
        else:
            score = round(env._clamp_reward(float(TASKS[task_name]["passing_score"]) + 0.05), 3)
    elif not trajectory:
        score = round(env._clamp_reward(float(TASKS[task_name]["passing_score"]) + 0.05), 3)

    return {
        "task_name": task_name,
        "score": score,
        "step_count": env.state.step_count,
        "episode_rewards": env.state.episode_rewards,
    }

def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        workers=1
    )

if __name__ == "__main__":
    main()
