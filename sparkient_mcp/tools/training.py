"""Model training tools."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from sparkient_mcp.server import mcp
from sparkient_mcp.client import get_client


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def train_model(
    decision_type_id: Annotated[
        str,
        Field(description="UUID of the decision type."),
    ],
    preset: Annotated[
        str,
        Field(description="Training preset: 'fast', 'balanced', or 'thorough'."),
    ] = "balanced",
    auto_deploy: Annotated[
        bool,
        Field(description="If true, deploy the trained model automatically."),
    ] = True,
) -> dict[str, Any]:
    """Trigger model training for a decision type.

    Training runs asynchronously. The pipeline:
      1. Extracts features from examples (numeric, categorical, text embeddings)
      2. Trains a decision classifier with automated optimization
      3. Compiles the model for fast inference
      4. Optionally deploys the model for live inference

    Available presets:
      - 'fast' — quick training, faster optimization
      - 'balanced' — default, good accuracy/speed trade-off
      - 'thorough' — maximum accuracy, more thorough optimization
    """
    client = get_client()
    return await client.train_model(decision_type_id, preset, auto_deploy)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_training_status(
    decision_type_id: Annotated[
        str,
        Field(description="UUID of the decision type."),
    ],
    policy_id: Annotated[
        str,
        Field(description="UUID of the training policy (returned by train_model)."),
    ],
) -> dict[str, Any]:
    """Check real-time training progress for a model being trained.

    Returns the current training stage, progress percentage, elapsed time,
    and a list of completed stages. Poll every 3-5 seconds while training
    is in progress.

    The training pipeline has multiple stages:
      1. Generating training data (if needed)
      2. Preparing and balancing data
      3. Analyzing input features
      4. Training text analysis model (if text fields present)
      5. Training decision classifier
      6. Evaluating model quality
      7. Finalizing and packaging
    """
    client = get_client()
    return await client.get_training_progress(decision_type_id, policy_id)
