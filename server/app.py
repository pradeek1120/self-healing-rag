import uvicorn
from openenv.core.env_server import create_app
from .environment import RAGEnvironment
from ..models import RAGAction, RAGObservation

app = create_app(RAGEnvironment, RAGAction, RAGObservation)

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
