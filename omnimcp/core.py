# omnimcp/core.py

from typing import List, Tuple, Optional
import platform

from omnimcp.types import (
    UIElement,
    ElementTrack,
    ActionDecision,
    LLMAnalysisAndDecision,
)
from omnimcp.utils import (
    render_prompt,
    logger,
)
from omnimcp.completions import call_llm_api
from omnimcp.config import config


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
      # Add other parameters as needed (e.g., scroll_direction, scroll_steps)
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

# --- Planner Function ---


def plan_action_for_ui(
    elements: List[UIElement],
    user_goal: str,
    action_history: List[str] | None = None,
    step: int = 0,
    tracking_info: Optional[List[ElementTrack]] = None,
) -> Tuple[ActionDecision, Optional[UIElement]]:  # Updated return type
    """
    Uses an LLM to analyze UI state with tracking and plan the next action.

    Args:
        elements: Raw UI elements detected in the current frame.
        user_goal: The overall goal description.
        action_history: Descriptions of previous actions taken.
        step: The current step number.
        tracking_info: List of ElementTrack objects from the tracker.

    Returns:
        A tuple containing the ActionDecision object from the LLM
        and the targeted UIElement (if any) found in the current frame.
    """
    action_history = action_history or []
    logger.info(
        f"Planning action for goal: '{user_goal}' with {len(elements)} raw elements. "
        f"History: {len(action_history)} steps. Tracking: {len(tracking_info or [])} active tracks."
    )

    # Limit elements and tracks passed to the prompt for performance/context window
    MAX_ELEMENTS_IN_PROMPT = 50
    MAX_TRACKS_IN_PROMPT = 50
    elements_for_prompt = elements[:MAX_ELEMENTS_IN_PROMPT]
    tracking_info_for_prompt = (
        tracking_info[:MAX_TRACKS_IN_PROMPT] if tracking_info else None
    )

    # Render the prompt using the template and current context
    prompt = render_prompt(
        PROMPT_TEMPLATE,
        user_goal=user_goal,
        elements=elements_for_prompt,
        action_history=action_history,
        platform=platform.system(),
        tracking_info=tracking_info_for_prompt,  # Include tracking info
    )

    # Define the system prompt guiding the LLM's output format
    system_prompt = (
        "You are an AI assistant. Respond ONLY with a single valid JSON object "
        "containing the keys 'screen_analysis' and 'action_decision', conforming "
        "to the specified Pydantic models. Do not include any explanatory text "
        "before or after the JSON block, and do not use markdown code fences like ```json."
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        # Call the LLM API expecting the combined analysis and decision structure
        llm_output: LLMAnalysisAndDecision = call_llm_api(
            messages,
            LLMAnalysisAndDecision,  # Expect the combined model for validation
            system_prompt=system_prompt,
            model=config.ANTHROPIC_DEFAULT_MODEL,  # Use model from config
        )
        # Log the structured analysis and decision for debugging purposes
        # Use model_dump_json for pretty printing if desired, or just log the object
        logger.debug(f"LLM Screen Analysis Received: {llm_output.screen_analysis}")
        logger.debug(f"LLM Action Decision Received: {llm_output.action_decision}")

    except (ValueError, Exception) as e:
        logger.error(
            f"Failed to get valid analysis/decision from LLM: {e}", exc_info=True
        )
        # Propagate the error to halt execution on planning failure
        raise

    # Extract the decision part to be returned
    decision = llm_output.action_decision

    # Find the target UIElement in the current frame based on the ID from the decision
    target_ui_element = None
    if decision.target_element_id is not None:
        # Search through the raw elements detected in *this* frame
        target_ui_element = next(
            (el for el in elements if el.id == decision.target_element_id), None
        )
        if target_ui_element is None:
            logger.warning(
                f"LLM targeted element ID {decision.target_element_id} in action decision, "
                f"but it was not found in the current raw elements list ({len(elements)} elements)."
            )
            # The target_ui_element remains None, AgentExecutor action handlers must check for this

    logger.info(
        f"Planner returning action_type: {decision.action_type}, "
        f"Target Elem ID: {decision.target_element_id}, "
        f"Goal Complete: {decision.is_goal_complete}"
    )

    # Return the validated ActionDecision object and the resolved target element
    return decision, target_ui_element
