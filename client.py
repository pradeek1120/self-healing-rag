"""
client.py
==========
Typed HTTP client for the Self-Healing RAG Environment.
Uses openenv-core's HTTPEnvClient base class.

Usage:
    from rag_env import RAGEnv, RAGAction

    with RAGEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_name="task_detect_hallucination")
        result = env.step(RAGAction(action_type="answer", content="20 days"))
        print(result.observation)
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import RAGAction, RAGObservation, RAGState
except ImportError:
    from models import RAGAction, RAGObservation, RAGState


class RAGEnv(EnvClient[RAGAction, RAGObservation, RAGState]):
    """
    Client for the Self-Healing RAG Environment.
    Handles all HTTP communication with the server.
    """

    def _step_payload(self, action: RAGAction) -> dict:
        """Convert action to dict for HTTP request."""
        return {
            "action_type": action.action_type,
            "content": action.content,
            "target_doc_id": action.target_doc_id,
        }

    def _parse_result(self, payload: dict) -> StepResult[RAGObservation]:
        """Parse HTTP response into typed StepResult."""
        obs = RAGObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> RAGState:
        """Parse state response."""
        return RAGState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            current_task=payload.get("current_task", ""),
            hallucination_detected=payload.get("hallucination_detected", False),
            database_fixed=payload.get("database_fixed", False),
            fix_log=payload.get("fix_log", []),
            episode_rewards=payload.get("episode_rewards", []),
            done=payload.get("done", False),
        )
