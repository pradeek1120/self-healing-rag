"""
models.py
==========
Defines the typed Action, Observation, and State classes.
These MUST use Pydantic and follow the openenv-core spec.

Think of these as:
- Action     = what the agent CAN DO
- Observation = what the agent SEES
- State      = internal environment bookkeeping
"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class RAGAction:
    """
    What the AI agent can do at each step.

    action_type options:
      "answer"      → Give an answer to the question
      "detect"      → Scan documents for conflicts/outdated versions
      "find_source" → Identify which document ID caused the hallucination
      "fix"         → Auto-fix: archive wrong doc, promote correct one
      "verify"      → Re-answer after fix to confirm it worked
    """
    action_type: str
    content: str
    target_doc_id: Optional[str] = None


@dataclass
class RAGObservation:
    """
    What the AI agent sees after each step.
    Like a dashboard showing current environment status.
    """
    question: str
    retrieved_documents: list
    current_answer: Optional[str] = None
    hallucination_detected: bool = False
    conflicting_docs: list = field(default_factory=list)
    database_fixed: bool = False
    step_number: int = 0
    message: str = ""
    reward: float = 0.0


@dataclass
class RAGState:
    """
    Internal state of the environment.
    Tracks episode progress.
    """
    episode_id: str = ""
    step_count: int = 0
    current_task: str = ""
    hallucination_detected: bool = False
    database_fixed: bool = False
    fix_log: list = field(default_factory=list)
    episode_rewards: list = field(default_factory=list)
    done: bool = False
