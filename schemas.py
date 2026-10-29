from typing import Literal
from pydantic import BaseModel, Field


class NextStep(BaseModel):
    """Pydantic schema representing the Supervisor's routing decision."""

    next_agent: Literal["Researcher", "Writer", "Editor", "Human_Approval", "FINISH"] = Field(
        ...,
        description="The next agent to execute in the workflow, or 'FINISH' if final approval has been granted."
    )
    reason: str = Field(
        ...,
        description="Detailed rationale and explanation for why this next agent or action was chosen."
    )
