import uvicorn
from fastapi import Body
from openenv.core.env_server import create_fastapi_app

try:
    from server.environment import MAX_SCORE, MIN_SCORE, RAGEnvironment
    from models import RAGAction, RAGObservation
    from tasks import TASKS, list_tasks
except ImportError:
    from .environment import MAX_SCORE, MIN_SCORE, RAGEnvironment
    from ..models import RAGAction, RAGObservation
    from ..tasks import TASKS, list_tasks


app = create_fastapi_app(
    RAGEnvironment,
    RAGAction,
    RAGObservation,
    max_concurrent_envs=8,
)
app.title = "self-healing-rag"
app.version = "1.1.0"
app.description = (
    "Self-healing RAG benchmark with randomized task instances, hidden "
    "ground-truth labels, and standard OpenEnv WebSocket/MCP support."
)


def _serialize_task(task):
    payload = dict(task.__dict__)
    payload["task_id"] = task.id
    payload["passing_score"] = _exclusive_score(task.passing_score)
    payload["score"] = payload["passing_score"]
    return payload


def _exclusive_score(value) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = MIN_SCORE
    return round(max(MIN_SCORE, min(numeric_value, MAX_SCORE)), 3)


@app.get("/")
def root():
    return {
        "name": "self-healing-rag",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "schema": "/schema",
        "tasks": "/tasks",
        "grader": "/grader",
        "ws": "/ws",
        "mcp": "/mcp",
    }


@app.get("/tasks")
def tasks():
    return {
        "tasks": [_serialize_task(task) for task in list_tasks()],
        "action_schema": RAGAction.model_json_schema(),
        "observation_schema": RAGObservation.model_json_schema(),
    }


@app.get("/info")
def info():
    return {
        "name": "self-healing-rag",
        "version": "1.1.0",
        "tasks": [_serialize_task(task) for task in list_tasks()],
        "action_schema": RAGAction.model_json_schema(),
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
    seed = (
        payload.get("seed")
        or payload.get("session_id")
        or payload.get("episode_id")
        or task_name
    )

    env = RAGEnvironment()
    env.reset(task_name=task_name, seed=seed)

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

    score = _exclusive_score(env.get_episode_score())
    if (not trajectory) and rewards:
        if isinstance(rewards, str):
            reward_values = [float(value) for value in rewards.split(",") if value.strip()]
        elif isinstance(rewards, list):
            reward_values = [float(value) for value in rewards]
        else:
            reward_values = []
        if reward_values:
            score = _exclusive_score(sum(reward_values) / len(reward_values))

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
        workers=1,
    )


if __name__ == "__main__":
    main()
