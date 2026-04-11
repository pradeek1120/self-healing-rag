from .environment import RAGEnvironment

__all__ = ["app", "RAGEnvironment"]


def __getattr__(name: str):
    if name == "app":
        from .app import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
