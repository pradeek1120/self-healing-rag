import uuid

import uvicorn
from fastapi import Body, FastAPI, HTTPException

try:
    from server.environment import RAGEnvironment
    from models import RAGAction, RAGObservation, RAGState
    from tasks import TASKS, list_tasks
except ImportError:
    from .environment import RAGEnvironment
    from ..models import RAGAction, RAGObservation, RAGState
    from ..tasks import TASKS, list_tasks


app = FastAPI(title="self-healing-rag")
_SESSIONS: dict[str, RAGEnvironment] = {}
DEFAULT_SESSION_ID = "default"


def _resolve_session_id(payload: dict) -> str:
    return (
        payload.get("session_id")
        or payload.get("episode_id")
        or payload.get("id")
        or DEFAULT_SESSION_ID
    )


def _serialize_observation(observation: RAGObservation, session_id: str) -> dict:
    obs_body = observation.model_dump(exclude={"reward", "done"})
    return {
        "session_id": session_id,
        "observation": obs_body,
        "reward": observation.reward,
        "done": observation.done,
        **observation.model_dump(),
    }


def _get_or_create_env(session_id: str) -> RAGEnvironment:
    env = _SESSIONS.get(session_id)
    if env is None:
        env = RAGEnvironment()
        _SESSIONS[session_id] = env
    return env


def _get_env_or_404(session_id: str) -> RAGEnvironment:
    env = _SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return env


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
        "tasks": "/tasks",
        "grader": "/grader",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "self-healing-rag",
        "description": "Self-healing RAG environment with explicit task scoring",
        "version": "1.0.0",
    }


@app.get("/schema")
def schema():
    return {
        "action": RAGAction.model_json_schema(),
        "observation": RAGObservation.model_json_schema(),
        "state": RAGState.model_json_schema(),
    }


@app.get("/tasks")
def tasks():
    return {
        "tasks": [task.__dict__ for task in list_tasks()],
        "action_schema": RAGAction.model_json_schema(),
        "observation_schema": RAGObservation.model_json_schema(),
        "state_schema": RAGState.model_json_schema(),
    }


@app.get("/info")
def info():
    return {
        "name": "self-healing-rag",
        "tasks": [task.__dict__ for task in list_tasks()],
        "action_schema": RAGAction.model_json_schema(),
    }


@app.post("/reset")
def reset(payload: dict = Body(default={})):
    task_name = payload.get("task_name") or payload.get("task_id") or "task_detect_hallucination"
    task_name = task_name if task_name in TASKS else "task_detect_hallucination"
    session_id = _resolve_session_id(payload)
    if session_id == DEFAULT_SESSION_ID and payload.get("session_id") is None:
        session_id = DEFAULT_SESSION_ID
    elif session_id == DEFAULT_SESSION_ID:
        session_id = str(uuid.uuid4())

    env = _get_or_create_env(session_id)
    observation = env.reset(task_name=task_name)
    return _serialize_observation(observation, session_id)


@app.post("/step")
def step(payload: dict = Body(default={})):
    session_id = _resolve_session_id(payload)
    env = _get_or_create_env(session_id)

    action_payload = payload.get("action", payload)
    if not isinstance(action_payload, dict):
        raise HTTPException(status_code=422, detail="Invalid action payload")

    action = RAGAction(
        action_type=action_payload.get("action_type", "answer"),
        content=action_payload.get("content", ""),
        target_doc_id=action_payload.get("target_doc_id"),
    )
    observation = env.step(action)
    return _serialize_observation(observation, session_id)


@app.post("/state")
def state(payload: dict = Body(default={})):
    session_id = _resolve_session_id(payload)
    env = _get_env_or_404(session_id)
    return {"session_id": session_id, **env.state.model_dump()}


@app.post("/grader")
def grader(payload: dict = Body(default={})):
    task_name = (
        payload.get("task_name")
        or payload.get("task_id")
        or payload.get("task")
        or "task_detect_hallucination"
    )
    task_name = task_name if task_name in TASKS else "task_detect_hallucination"
    session_id = _resolve_session_id(payload)
    env = _SESSIONS.get(session_id)

    if env is not None:
        score = env.get_episode_score()
        return {
            "task_name": task_name,
            "session_id": session_id,
            "score": score,
            "step_count": env.state.step_count,
            "episode_rewards": env.state.episode_rewards,
        }

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

    return {
        "task_name": task_name,
        "session_id": session_id,
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
