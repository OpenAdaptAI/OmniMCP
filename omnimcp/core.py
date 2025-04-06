# omnimcp/core.py
from typing import List, Tuple, Optional  # Added Dict, Any
import platform

# Import necessary types
from .types import (
    UIElement,
    ElementTrack,  # Added
    LLMActionPlan,  # Still needed for temporary return value
    LLMAnalysisAndDecision,  # Added
)
from .utils import (
    render_prompt,
    logger,
)
from .completions import call_llm_api
from .config import config  # Import config if needed, e.g., for model name


# --- Updated Prompt Template ---
PROMPT_TEMPLATE = """
You are an expert UI automation assistant. Your task is to analyze the current UI state, including changes from the previous step, and then decide the single best next action to achieve a given goal.

**Operating System:** {{ platform }}

**User Goal:**
{{ user_goal }}

**Previous Actions Taken (up to last 5):**
{% if action_history %}
{% for action_desc in action_history[-5:] %} {# Show only recent history #}
- {{ action_desc }}
{% endfor %}
{% else %}
- None
{% endif %}

**Current UI Elements (Raw Detections - Max 50):**
```
{% for element in elements[:50] %}
{{ element.to_prompt_repr() }} {# Uses per-frame ID #}
{% endfor %}
```

**Tracked Elements Context (Persistent View - Max 50):**
This shows elements being tracked across frames. Status 'VISIBLE' means seen this frame. 'MISSING(n)' means missed for n consecutive frames.
```
{% if tracking_info %}
{% for track in tracking_info[:50] %}
- {{ track.short_repr() }} {# Uses persistent TrackID and status #}
{% endfor %}
{% else %}
- (No tracking info available or first frame)
{% endif %}
```

**Instructions:**

1.  **Analyze State:** Carefully review the Goal, History, Raw Elements, and especially the Tracked Elements Context. Reason about what changed since the last step (newly appeared elements? previously visible elements now missing? critical elements still present?). Consider if missing elements are temporary (e.g., due to UI transition) or permanent. Note any critical elements needed for the goal and their current status.
2.  **Decide Action:** Based on your analysis, determine the single best action to take next towards the goal. This could be interacting with a visible element, handling a missing element (e.g., waiting, using a keyboard shortcut if applicable), or finishing if the goal is complete.
3.  **Output Format:** Respond ONLY with a single valid JSON object containing two keys: "screen_analysis" and "action_decision".
    * The value for "screen_analysis" MUST be a JSON object conforming to the `ScreenAnalysis` structure below.
    * The value for "action_decision" MUST be a JSON object conforming to the `ActionDecision` structure below.
    * Do NOT include any text outside this main JSON object (e.g., no ```json markdown).

**JSON Output Structure:**

```json
{
  "screen_analysis": {
    "reasoning": "Your detailed step-by-step analysis of the current state, changes from the previous state using tracking context, and assessment relevant to the goal.",
    "disappeared_elements": ["list", "of", "track_ids", "considered", "permanently", "gone"],
    "temporarily_missing_elements": ["list", "of", "track_ids", "likely", "to", "reappear"],
    "new_elements": ["list", "of", "track_ids", "for", "newly", "appeared", "elements"],
    "critical_elements_status": {
      "track_id_example_1": "Visible",
      "track_id_example_2": "Missing"
    }
  },
  "action_decision": {
    "analysis_reasoning": "Brief summary connecting the screen analysis to the chosen action.",
    "action_type": "click | type | scroll | press_key | wait | finish",
    "target_element_id": <CURRENT_FRAME_ID_of_target_element_if_visible_and_applicable>,
    "parameters": {
      "text_to_type": "<text_if_action_is_type>",
      "key_info": "<key_if_action_is_press_key>",
      "wait_duration_s": <seconds_if_action_is_wait>
      # Add other parameters as needed
    },
    "is_goal_complete": <true_if_goal_is_fully_achieved_else_false>
  }
}

```

**Action Rules (Apply to `action_decision` fields):**
* If `action_type` is 'click', `target_element_id` MUST be the integer ID (from Current UI Elements) of a visible element. `parameters` should be empty or contain only non-essential info like `click_type`.
* If `action_type` is 'type', `parameters.text_to_type` MUST be the string to type. `target_element_id` SHOULD be the ID of the target field if identifiable.
* If `action_type` is 'press_key', `parameters.key_info` MUST be the key/shortcut string. `target_element_id` MUST be null.
* If `action_type` is 'scroll', specify direction/amount in `analysis_reasoning` or `parameters` if possible. `target_element_id` MUST be null.
* If `action_type` is 'wait', specify `parameters.wait_duration_s`. `target_element_id` MUST be null.
* If `action_type` is 'finish', `is_goal_complete` MUST be true. `target_element_id` and `parameters` should generally be null/empty.
* If a required element is missing (use Tracked Elements Context), choose an appropriate action like 'wait' or 'press_key' if a keyboard alternative exists, or explain the issue in `screen_analysis.reasoning` and potentially choose 'finish' with `is_goal_complete: false` if stuck. Do NOT hallucinate `target_element_id` for missing elements.
"""

# --- Updated Planner Function ---


def plan_action_for_ui(
    elements: List[UIElement],
    user_goal: str,
    action_history: List[str] | None = None,
    step: int = 0,
    tracking_info: Optional[List[ElementTrack]] = None,  # Accept list of ElementTrack
) -> Tuple[
    LLMActionPlan, Optional[UIElement]
]:  # Still return LLMActionPlan temporarily
    """
    Uses an LLM to analyze UI state with tracking and plan the next action.

    Args:
        elements: Raw UI elements detected in the current frame.
        user_goal: The overall goal description.
        action_history: Descriptions of previous actions taken.
        step: The current step number.
        tracking_info: List of ElementTrack objects from the tracker.

    Returns:
        A tuple containing an LLMActionPlan (converted from ActionDecision)
        and the targeted UIElement (if any) found in the current frame.
    """
    action_history = action_history or []
    logger.info(
        f"Planning action for goal: '{user_goal}' with {len(elements)} raw elements. "
        f"History: {len(action_history)} steps. Tracking: {len(tracking_info or [])} active tracks."
    )

    # Limit elements and tracks passed to the prompt for brevity
    MAX_ELEMENTS_IN_PROMPT = 50
    MAX_TRACKS_IN_PROMPT = 50
    elements_for_prompt = elements[:MAX_ELEMENTS_IN_PROMPT]
    tracking_info_for_prompt = (
        tracking_info[:MAX_TRACKS_IN_PROMPT] if tracking_info else None
    )

    prompt = render_prompt(
        PROMPT_TEMPLATE,
        user_goal=user_goal,
        elements=elements_for_prompt,
        action_history=action_history,
        platform=platform.system(),
        tracking_info=tracking_info_for_prompt,  # Pass tracking info
    )

    # System prompt reinforcing the JSON structure
    system_prompt = (
        "You are an AI assistant. Respond ONLY with a single valid JSON object "
        "containing the keys 'screen_analysis' and 'action_decision', conforming "
        "to the specified Pydantic models. Do not include any explanatory text "
        "before or after the JSON block, and do not use markdown code fences like ```json."
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        # Call LLM expecting the combined analysis and decision structure
        llm_output = call_llm_api(
            messages,
            LLMAnalysisAndDecision,  # Expect the combined model
            system_prompt=system_prompt,
            model=config.ANTHROPIC_DEFAULT_MODEL,  # Use configured model
        )
        # Log the structured analysis and decision for debugging
        logger.debug(
            f"LLM Screen Analysis: {llm_output.screen_analysis.model_dump_json(indent=2)}"
        )
        logger.debug(
            f"LLM Action Decision: {llm_output.action_decision.model_dump_json(indent=2)}"
        )

    except (ValueError, Exception) as e:
        logger.error(
            f"Failed to get valid analysis/decision from LLM: {e}", exc_info=True
        )
        # Fallback or re-raise? Re-raise for now to halt execution on planning failure.
        raise

    # --- Temporary Conversion back to LLMActionPlan ---
    # This allows AgentExecutor handlers to work without immediate refactoring.
    # TODO: Refactor AgentExecutor later to consume ActionDecision directly.
    analysis = llm_output.screen_analysis
    decision = llm_output.action_decision

    # Combine reasoning (can be refined)
    combined_reasoning = f"Analysis: {analysis.reasoning}\nDecision Justification: {decision.analysis_reasoning}"

    # Extract parameters for LLMActionPlan
    # Ensure parameters is not None before accessing .get()
    parameters = decision.parameters or {}
    text_param = parameters.get("text_to_type")
    key_param = parameters.get("key_info")
    # Add handling for 'wait' action type if needed by LLMActionPlan later
    # wait_param = parameters.get("wait_duration_s")

    # Handle potential new action types like 'wait' or 'finish' if LLMActionPlan
    # doesn't support them directly yet. For now, map 'finish'/'wait' to a state?
    # Let's assume LLMActionPlan.action can hold the new types for now.
    action_type = decision.action_type

    converted_plan = LLMActionPlan(
        reasoning=combined_reasoning,
        action=action_type,  # Pass action_type directly
        element_id=decision.target_element_id,  # Pass the current frame ID
        text_to_type=text_param,
        key_info=key_param,
        is_goal_complete=decision.is_goal_complete,
    )
    # Validate the converted plan (optional, but good practice)
    try:
        # Re-validate the object created from ActionDecision fields
        # This ensures the LLM followed rules that map to LLMActionPlan
        # Note: This validation might fail if action_type is 'wait' or 'finish'
        # We might need to adjust LLMActionPlan or skip validation for new types.
        # For now, let's try validating.
        LLMActionPlan.model_validate(converted_plan.model_dump())
    except Exception as validation_err:
        logger.warning(
            f"Converted LLMActionPlan failed validation (potentially due to new action types like '{action_type}'): {validation_err}"
        )
        # Don't raise, just warn for now, as the ActionDecision was likely valid.

    # Find the target UIElement based on the element_id from the decision
    target_ui_element = None
    if converted_plan.element_id is not None:
        target_ui_element = next(
            (el for el in elements if el.id == converted_plan.element_id), None
        )
        if target_ui_element is None:
            logger.warning(
                f"LLM targeted element ID {converted_plan.element_id}, but it was not found in the current raw elements."
            )
            # Keep element_id in plan, but target_ui_element remains None

    logger.info(
        f"Planner returning action: {converted_plan.action}, Target Elem ID: {converted_plan.element_id}, Goal Complete: {converted_plan.is_goal_complete}"
    )

    return converted_plan, target_ui_element  # Return converted plan and element
