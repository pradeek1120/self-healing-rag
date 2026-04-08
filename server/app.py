import uvicorn
from openenv.core.env_server import create_app

try:
    from server.environment import RAGEnvironment
    from models import RAGAction, RAGObservation
except ImportError:
    from .environment import RAGEnvironment
    from ..models import RAGAction, RAGObservation

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

def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        workers=1
    )

if __name__ == "__main__":
    main()
