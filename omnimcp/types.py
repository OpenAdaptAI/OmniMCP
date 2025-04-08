# omnimcp/types.py

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from pydantic_prompt import prompt_schema

# Define Bounds (assuming normalized coordinates 0.0-1.0)
Bounds = Tuple[float, float, float, float]  # (x, y, width, height)


# --- Core Data Structures (Using Dataclasses as provided) ---


@dataclass
class UIElement:
    """Represents a UI element detected in a single frame."""

    # Per-frame ID assigned by parser/mapper
    id: int
    type: str  # button, text_field, checkbox, link, text, etc.
    content: str  # Text content or accessibility label
    bounds: Bounds  # Normalized coordinates (x, y, width, height)
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)  # e.g., {'checked': False}

    def to_dict(self) -> Dict[str, Any]:
        """Convert UIElement to a dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "bounds": self.bounds,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }

    def to_prompt_repr(self) -> str:
        """Concise string representation suitable for LLM prompts."""
        bound_str = (
            f"({self.bounds[0]:.3f}, {self.bounds[1]:.3f}, "
            f"{self.bounds[2]:.3f}, {self.bounds[3]:.3f})"
        )
        content_preview = (
            (self.content[:30] + "...") if len(self.content) > 33 else self.content
        )
        # Avoid newlines in prompt list
        content_preview = content_preview.replace("\n", " ")
        type_lower = self.type.lower() if isinstance(self.type, str) else "unknown"
        return (
            f"ID: {self.id}, Type: {type_lower}, "
            f"Content: '{content_preview}', Bounds: {bound_str}"
        )

    def short_repr(self) -> str:
        """Provides a short representation using the per-frame ID."""
        content_preview = self.content[:25].replace("\n", " ")
        if len(self.content) > 25:
            content_preview += "..."
        type_lower = self.type.lower() if isinstance(self.type, str) else "unknown"
        return f"ID {self.id} ({type_lower} '{content_preview}')"


@dataclass
class ScreenState:
    """Represents the raw state of the screen at a point in time."""

    elements: List[UIElement]
    dimensions: Tuple[int, int]  # Actual pixel dimensions
    timestamp: float


# --- Action / Interaction Results (Using Dataclasses) ---


@dataclass
class ActionVerification:
    """Optional verification data for an action's effect."""

    success: bool
    before_state: bytes  # Screenshot bytes
    after_state: bytes  # Screenshot bytes
    changes_detected: List[Bounds]  # Regions where changes occurred
    confidence: float


@dataclass
class InteractionResult:
    """Generic result of an interaction attempt."""

    success: bool
    element: Optional[UIElement]  # The element interacted with, if applicable
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    verification: Optional[ActionVerification] = None


@dataclass
class ScrollResult(InteractionResult):
    """Result specific to a scroll action."""

    scroll_amount: float = 0.0


@dataclass
class TypeResult(InteractionResult):
    """Result specific to typing text."""

    text_entered: str = ""


# --- Error / Debug Context (Using Dataclasses) ---


@dataclass
class ToolError:
    """Rich error information, potentially for MCP tools or agent errors."""

    message: str
    visual_context: Optional[bytes]  # Screenshot bytes
    attempted_action: str
    element_description: str  # Description or ID of intended target
    recovery_suggestions: List[str]


@dataclass
class DebugContext:
    """Context for debugging a specific operation or tool call."""

    tool_name: str
    inputs: Dict[str, Any]
    result: Any
    duration: float
    visual_state: Optional[ScreenState]  # Raw screen state at the time
    error: Optional[Dict] = None  # e.g., ToolError as dict

    def save_snapshot(self, path: str) -> None:
        """Save debug snapshot for analysis."""
        # Implementation would involve serializing state/context to a file
        logger.warning("DebugContext.save_snapshot not yet implemented.")


# --- LLM Plan / Action Structures (Using Pydantic for Validation) ---


class LLMActionPlan(BaseModel):
    """
    Defines the structured output expected from the LLM for basic action planning.
    This might be superseded by ActionDecision but serves as the current target.
    """

    reasoning: str = Field(
        ..., description="Step-by-step thinking process leading to the chosen action."
    )
    action: Literal["click", "type", "scroll", "press_key", "wait"] = Field(
        ..., description="The single next action to perform."
    )
    is_goal_complete: bool = Field(
        ...,
        description="Set to true if the user's overall goal is fully achieved by the current state, false otherwise.",
    )
    element_id: Optional[int] = Field(
        default=None,
        description="The per-frame ID of the target UI element IF the action is 'click' or 'type' and goal is not complete. Must be null otherwise.",
    )
    text_to_type: Optional[str] = Field(
        default=None,
        description="Text to type IF action is 'type' and goal is not complete. Must be null otherwise.",
    )
    key_info: Optional[str] = Field(
        default=None,
        description="Key or shortcut to press IF action is 'press_key' and goal is not complete (e.g., 'Enter', 'Cmd+Space'). Must be null otherwise.",
    )

    @field_validator("element_id")
    @classmethod
    def check_element_id(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        # Skip validation if goal is already complete
        if info.data.get("is_goal_complete", False):
            return v

        action = info.data.get("action")
        if action == "click" and v is None:
            raise ValueError(
                "element_id is required for action 'click' when goal is not complete"
            )
        if action in ["scroll", "press_key"] and v is not None:
            raise ValueError(
                f"element_id must be null for action '{action}' when goal is not complete"
            )
        return v

    @field_validator("text_to_type")
    @classmethod
    def check_text_to_type(
        cls, v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        if info.data.get("is_goal_complete", False):
            return v
        action = info.data.get("action")
        if action == "type" and v is None:
            raise ValueError(
                "text_to_type (even empty string) is required for action 'type' when goal is not complete"
            )
        if action != "type" and v is not None:
            raise ValueError(
                "text_to_type must be null for actions other than 'type' when goal is not complete"
            )
        return v

    @field_validator("key_info")
    @classmethod
    def check_key_info(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        if info.data.get("is_goal_complete", False):
            return v
        action = info.data.get("action")
        if action == "press_key" and v is None:
            raise ValueError(
                "key_info is required for action 'press_key' when goal is not complete"
            )
        if action != "press_key" and v is not None:
            raise ValueError(
                "key_info must be null for actions other than 'press_key' when goal is not complete"
            )
        return v


# --- Models for Tracking and Advanced Planning (Issue #8) ---


class ElementTrack(BaseModel):
    """Tracking information for a UI element across frames, managed by SimpleElementTracker."""

    track_id: str = Field(
        description="Persistent tracking ID assigned by the tracker (e.g., 'track_0')"
    )
    # Storing Optional[UIElement] (dataclass) directly in Pydantic model works
    latest_element: Optional[UIElement] = Field(
        None,
        description="The UIElement dataclass instance detected in the current frame, if any.",
    )
    consecutive_misses: int = Field(
        0,
        description="Number of consecutive frames this element track was not detected.",
    )
    last_seen_frame: int = Field(
        0,
        description="The frame number when this track was last successfully detected.",
    )

    def short_repr(self) -> str:
        """Short representation for prompt, using persistent track_id."""
        status = (
            "VISIBLE" if self.latest_element else f"MISSING({self.consecutive_misses})"
        )
        if self.latest_element:
            # Use the short_repr from the underlying UIElement dataclass
            element_repr = self.latest_element.short_repr()  # Gets ID, type, content
            return f"TrackID {self.track_id} [{element_repr}] - Status: {status}, LastSeen: f{self.last_seen_frame}"
        else:
            # If missing, we don't know the type/content from this object alone
            return f"TrackID {self.track_id} (Type Unknown) - Status: {status}, LastSeen: f{self.last_seen_frame}"


@prompt_schema
class ScreenAnalysis(BaseModel):
    """LLM's analysis of the current UI state with tracking information."""

    reasoning: str
    """Detailed reasoning about the UI state, changes from the previous state using tracking context, and assessment relevant to the goal."""

    disappeared_elements: List[str] = Field(default_factory=list)
    """List of track_ids considered permanently gone since the last visible frame."""

    temporarily_missing_elements: List[str] = Field(default_factory=list)
    """List of track_ids considered temporarily missing (e.g., due to UI transition) but likely to reappear."""

    new_elements: List[str] = Field(default_factory=list)
    """List of track_ids for newly appeared elements this frame."""

    critical_elements_status: Dict[str, str] = Field(default_factory=dict)
    """Dictionary mapping track_ids of elements deemed critical for the current goal/step to their status (e.g., 'Visible', 'Missing', 'Gone')."""


@prompt_schema
class ActionDecision(BaseModel):
    """LLM's decision on the next action based on its analysis."""

    analysis_reasoning: str
    """Brief summary connecting the screen analysis to the chosen action."""

    action_type: Literal["click", "type", "scroll", "press_key", "wait", "finish"]
    """The type of action to perform."""

    target_element_id: Optional[int] = Field(default=None)
    """The CURRENT per-frame 'id' of the target UIElement, IF the action applies to a specific visible element (e.g., 'click', 'type'). Must be null otherwise."""

    parameters: Dict[str, Any] = Field(default_factory=dict)
    """Action-specific parameters. Examples:
       - For 'type': {'text_to_type': 'string'}
       - For 'press_key': {'key_info': 'key_string'}
       - For 'wait': {'wait_duration_s': float}
       - For 'scroll': {'scroll_direction': 'up'/'down'/'left'/'right', 'scroll_steps': int}
       - For 'click': {'click_type': 'single'/'double'} (Optional)
    """

    is_goal_complete: bool = Field(default=False)
    """Set to true if the overall user goal is fully achieved after this action decision."""


# --- Model for Structured Step Logging ---


class LoggedStep(BaseModel):
    """Structure for logging data for a single agent step."""

    step_index: int
    timestamp: float = Field(default_factory=time.time)
    goal: str
    screenshot_path: Optional[str] = None  # Relative path within run dir

    # Inputs to Planner
    input_elements_count: int
    # Store list of dicts for JSON serialization compatibility
    tracking_context: Optional[List[Dict]] = Field(
        None, description="Snapshot of ElementTrack data (as dicts) provided to LLM"
    )
    action_history_at_step: List[str]

    # Planner Outputs (Store as dicts)
    llm_analysis: Optional[Dict] = Field(
        None, description="ScreenAnalysis output from LLM"
    )
    llm_decision: Optional[Dict] = Field(
        None, description="ActionDecision output from LLM"
    )
    raw_llm_action_plan: Optional[Dict] = Field(
        None, description="LLMActionPlan if ActionDecision not yet implemented"
    )

    # Execution
    executed_action: str
    executed_target_element_id: Optional[int] = None
    executed_parameters: Dict[str, Any]
    action_success: bool

    # Metrics
    perception_time_s: float
    planning_time_s: float
    execution_time_s: float
    step_time_s: float


@prompt_schema
class LLMAnalysisAndDecision(BaseModel):
    """Defines the full structured output expected from the LLM, combining analysis and decision."""

    screen_analysis: ScreenAnalysis
    """The LLM's analysis of the current screen state and element tracks."""

    action_decision: ActionDecision
    """The LLM's decision on the next action based on the analysis."""
