"""Model training tools."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from sparkient_mcp.client import get_client
from sparkient_mcp.server import mcp


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

    Training uses the complete text-model and classifier pipeline. Duration is
    workload-dependent; use get_training_status to follow the live run.
    """
    client = get_client()
    return await client.train_model(decision_type_id, auto_deploy)


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

    Returns the durable attempt identity, retry count, heartbeat, current
    stage, progress percentage, and completed stages. Poll every 3-5 seconds
    while status is ``training``, ``stopping``, or ``cancelling``. ``stopping``
    retains the organisation slot until the exact stale execution is terminal.

    The training pipeline has multiple stages:
      1. Generating training data (if needed)
      2. Preparing and balancing data
      3. Analyzing input features
      4. Training the task-specific text analysis model
      5. Training decision classifier
      6. Evaluating model quality
      7. Finalizing and packaging
    """
    client = get_client()
    return await client.get_training_progress(decision_type_id, policy_id)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def cancel_training(
    decision_type_id: Annotated[
        str,
        Field(description="UUID of the decision type."),
    ],
    policy_id: Annotated[
        str,
        Field(description="UUID of the active training policy."),
    ],
) -> dict[str, Any]:
    """Safely cancel one active training run.

    Sparkient first fences the exact policy attempt, then asks the compute
    platform to stop it. The response is ``cancelled`` after the stop is
    confirmed, or remains ``cancelling`` when confirmation is still pending.
    Repeating the call for an already-cancelled policy is safe.
    """
    client = get_client()
    return await client.cancel_training(decision_type_id, policy_id)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def retry_training(
    decision_type_id: Annotated[str, Field(description="UUID of the decision type.")],
    policy_id: Annotated[str, Field(description="UUID of the failed, retryable policy.")],
) -> dict[str, Any]:
    """Retry a recoverable failure against the same immutable snapshot.

    Safe checkpoints and the policy version are reused, and a successful retry
    cannot create a second completion charge.
    """
    client = get_client()
    return await client.retry_training(decision_type_id, policy_id)
