from typing import Optional, List, Any
from pydantic import BaseModel, Field

class RAGAction(BaseModel):
    action_type: str
    content: str
    target_doc_id: Optional[str] = None

class RAGObservation(BaseModel):
    question: str
    retrieved_documents: List[Any] = []
    current_answer: Optional[str] = None
    hallucination_detected: bool = False
    conflicting_docs: List[Any] = []
    database_fixed: bool = False
    step_number: int = 0
    message: str = ""
    reward: float = 0.0

class RAGState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    current_task: str = ""
    hallucination_detected: bool = False
    database_fixed: bool = False
    fix_log: List[Any] = []
    episode_rewards: List[float] = []
    done: bool = False
