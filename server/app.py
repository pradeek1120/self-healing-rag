"""
server/app.py
==============
Creates the FastAPI server using openenv-core's helper.
Exposes reset(), step(), state() over HTTP.
"""

import uvicorn
from openenv.core.env_server import create_app
from .environment import RAGEnvironment
from ..models import RAGAction, RAGObservation

# Create one shared environment instance
env = RAGEnvironment()

# Create the FastAPI app
app = create_app(env, RAGAction, RAGObservation)


def main():
    """Entry point required by openenv validate."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
