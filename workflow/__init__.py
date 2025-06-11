"""
Workflow package for the RAG-based storyboard system.
"""
from workflow.graph import create_workflow
from workflow.state import WorkflowState

__all__ = ["create_workflow", "WorkflowState"]

import json
from pydantic import BaseModel

class PydanticEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Pydantic models."""
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)

# Monkey-patch json to use our encoder
_original_dumps = json.dumps

def patched_dumps(*args, **kwargs):
    kwargs.setdefault('cls', PydanticEncoder)
    return _original_dumps(*args, **kwargs)

json.dumps = patched_dumps 