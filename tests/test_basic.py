
from pathlib import Path

from exercise_3.config import ENV_PATH, PROJECT_ROOT


def test_project_root_configuration():
    """The flattened layout resolves configuration from the repository root."""
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert ENV_PATH == PROJECT_ROOT / ".env"


def test_agentscope_api():
    """The pinned AgentScope release exposes the API used by the examples."""
    from agentscope.agent import AgentBase

    assert AgentBase.__name__ == "AgentBase"
