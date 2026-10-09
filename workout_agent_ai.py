import json
import os
from contextlib import redirect_stdout
from io import StringIO

from google import genai
from google.genai import types


# Import the scheduling tool without showing its sample output.
with redirect_stdout(StringIO()):
    from workout_agent import reschedule_workouts


MODEL = "gemini-3.5-flash-lite"
MAX_ITERATIONS = 2


CONSTRAINTS = [
    "Schedule no more than one leg-focused workout per week.",
    "Never schedule leg workouts on consecutive days.",
    "Preserve rest and recovery when possible.",
    "Never schedule a workout on an unavailable day.",
    "When a workout is missed, reconsider the remaining schedule instead of blindly adding it.",
    "Keep the workout focus lower-body/glute focused.",
    "Keep upper-body workouts light or moderate.",
    "Cardio is flexible.",
]


def _response_field(value, name):
    if isinstance(value, dict):
        return value.get(name)

    return getattr(value, name, None)


def _extract_response_text(response):
    text = _response_field(response, "output_text")

    if isinstance(text, str) and text.strip():
        return text.strip()

    parts = []

    for step in _response_field(response, "steps") or []:
        if _response_field(step, "type") != "model_output":
            continue

        for content in _response_field(step, "content") or []:
            text_part = _response_field(content, "text")

            if isinstance(text_part, str) and text_part.strip():
                parts.append(text_part.strip())

    return "\n".join(parts)


def _build_effective_schedule(schedule, missed_workouts):
    """
    The planned schedule contains workouts that were planned.
    A workout in missed_workouts was NOT completed.

    Remove missed workouts from the effective schedule shown
    to the reasoning model.
    """

    effective_schedule = schedule.copy()

    for day in missed_workouts:
        effective_schedule.pop(day, None)

    return effective_schedule


def _request_decision(
    client,
    situation,
    tool_result=None,
):
    # Initial decision
    if tool_result is None:

        # Important: when a workout was missed, the model must
        # use the scheduling tool rather than deciding feasibility itself.
        missed_workouts = situation.get("missed_workouts", [])

        if missed_workouts:
            decision_rule = (
                "A workout is listed in missed_workouts, so that workout "
                "was NOT completed. You MUST choose action='reschedule' "
                "so the verified Python scheduling tool can determine "
                "whether a feasible day exists. Do NOT choose keep for "
                "a missed workout before the tool checks the schedule."
            )
        else:
            decision_rule = (
                "Decide whether the schedule needs adjustment based on "
                "the current situation and constraints."
            )

        prompt = (
            "You are the reasoning layer of a workout scheduling agent.\n\n"

            "IMPORTANT STATE RULES:\n"
            "1. planned_schedule contains workouts that were planned, "
            "not necessarily completed.\n"
            "2. A day listed in missed_workouts means that the workout "
            "planned for that day was NOT completed.\n"
            "3. A missed workout does NOT count as a completed workout "
            "toward weekly limits.\n"
            "4. effective_schedule_for_decision already removes missed "
            "workouts from the planned schedule.\n"
            "5. Do not create or rewrite the schedule yourself. "
            "A separate verified Python tool performs schedule changes.\n\n"

            f"{decision_rule}\n\n"

            "Return ONLY a JSON object with exactly these fields:\n"
            '{"action":"reschedule" or "keep","reason":"brief explanation"}\n\n'

            f"Situation: {json.dumps(situation)}"
        )

    # Follow-up decision after a tool call
    else:

        prompt = (
            "You are the reasoning layer of a workout scheduling agent.\n\n"

            "Review the result returned by the verified scheduling tool "
            "and decide what should happen next.\n\n"

            "IMPORTANT STATE RULES:\n"
            "1. A workout listed in missed_workouts was NOT completed.\n"
            "2. A missed workout does NOT count as a completed workout "
            "toward weekly limits.\n"
            "3. The scheduling tool is authoritative about what changes "
            "are actually possible.\n"
            "4. If the tool says no available day could be found, "
            "the task is complete. Choose keep and explain that the "
            "workout cannot be rescheduled this week.\n"
            "5. Do not create or rewrite a schedule yourself.\n\n"

            "Return ONLY a JSON object with exactly these fields:\n"
            '{"action":"reschedule" or "keep","reason":"brief explanation"}\n\n'

            f"Current situation: {json.dumps(situation)}\n"
            f"Previous tool result: {json.dumps(tool_result)}"
        )

    try:
        response = client.interactions.create(
            model=MODEL,
            input=prompt,
            generation_config={
                "thinking_level": "minimal"
            },
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["reschedule", "keep"],
                        },
                        "reason": {
                            "type": "string",
                        },
                    },
                    "required": ["action", "reason"],
                },
            },
            timeout=15,
        )

    except Exception as error:
        raise RuntimeError(
            f"Gemini request failed or timed out: {error}"
        ) from None

    response_text = _extract_response_text(response)

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    try:
        result = json.loads(response_text)

    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            f"Gemini returned invalid JSON: {response_text}"
        ) from error

    if (
        not isinstance(result, dict)
        or result.get("action") not in {"reschedule", "keep"}
    ):
        raise ValueError(
            'Gemini JSON must contain action "reschedule" or "keep".'
        )

    if (
        not isinstance(result.get("reason"), str)
        or not result["reason"].strip()
    ):
        raise ValueError(
            "Gemini JSON must contain a non-empty reason."
        )

    return {
        "action": result["action"],
        "reason": result["reason"].strip(),
    }


def _is_no_feasible_day(tool_result):
    """
    Detect the scheduler's terminal result:
    no available day exists for the requested workout.
    """

    if not isinstance(tool_result, dict):
        return False

    changes = tool_result.get("changes", [])

    if not isinstance(changes, list):
        return False

    return any(
        isinstance(change, str)
        and "Could not find an available day" in change
        for change in changes
    )


def workout_agent(
    schedule,
    missed_workouts=None,
    unavailable_days=None,
):
    missed_workouts = missed_workouts or []
    unavailable_days = unavailable_days or []

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    # One API attempt only to avoid repeated delays.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=15000,
            retry_options=types.HttpRetryOptions(
                attempts=1
            ),
        ),
    )

    updated_schedule = schedule.copy()
    pending_missed_workouts = list(missed_workouts)

    tool_result = None
    final_response = None
    changes = []

    for iteration in range(MAX_ITERATIONS):

        effective_schedule = _build_effective_schedule(
            updated_schedule,
            pending_missed_workouts,
        )

        situation = {
            "planned_schedule": updated_schedule,
            "effective_schedule_for_decision": effective_schedule,
            "missed_workouts": pending_missed_workouts,
            "unavailable_days": unavailable_days,
            "constraints": CONSTRAINTS,
        }

        result = _request_decision(
            client,
            situation,
            tool_result,
        )

        action = result["action"]

        print(
            f"Agent decision (iteration {iteration + 1}): {action}"
        )

        print(
            f"Agent reason: {result['reason']}"
        )

        if action == "keep":

            final_response = result["reason"]

            print("Tool called: no")

            if tool_result is None:
                print(
                    "Tool result: not applicable; "
                    "the agent kept the schedule unchanged."
                )
            else:
                print(
                    "Tool result: no additional call; "
                    "the prior result was retained."
                )

            break

        print("Tool called: reschedule_workouts")

        updated_schedule, changes = reschedule_workouts(
            updated_schedule,
            missed_workouts=pending_missed_workouts,
            unavailable_days=unavailable_days,
        )

        # The tool has processed the missed workout request.
        pending_missed_workouts = []

        tool_result = {
            "updated_schedule": updated_schedule,
            "changes": changes,
        }

        print("Tool result:")

        if changes:
            for change in changes:
                print(f"- {change}")
        else:
            print("- The scheduler made no changes.")

        # If the verified tool says no feasible day exists,
        # stop immediately. Do not waste another Gemini call.
        if _is_no_feasible_day(tool_result):

            final_response = (
                "No feasible day was available to reschedule the "
                "missed workout, so it must be dropped for this week."
            )

            print("Agent decision: stop")

            print(
                "Agent reason: The scheduling tool found no feasible day."
            )

            print(
                "Tool called: no additional call; "
                "the tool result is a terminal condition."
            )

            break

        # Do not make an unnecessary third call.
        if iteration == MAX_ITERATIONS - 1:

            final_response = (
                "The iteration limit was reached after the latest "
                "scheduling adjustment."
            )

            print("Agent decision: stop")

            print(
                "Agent reason: The maximum number of iterations was reached."
            )

            print(
                "Tool called: no additional call; "
                "the iteration limit was reached."
            )

            break

    print(
        f"Final agent response: {final_response}"
    )

    print("Final updated schedule:")

    for day, workout in updated_schedule.items():
        print(f"{day}: {workout}")

    return updated_schedule, changes



if __name__ == "__main__":
    scenarios = [
        {
            "name": "TEST 1: Missed Workout",
            "schedule": {
                "Monday": "Upper Body",
                "Tuesday": "Cardio",
                "Wednesday": "Upper Body",
                "Thursday": "Cardio",
                "Friday": "Upper Body",
                "Saturday": "Legs",
            },
            "missed": ["Saturday"],
            "unavailable": [],
        },
        {
            "name": "TEST 2: Unavailable Day",
            "schedule": {
                "Monday": "Upper Body",
                "Tuesday": "Cardio",
                "Wednesday": "Cardio",
                "Thursday": "Upper Body",
                "Friday": "Rest",
                "Saturday": "Legs",
            },
            "missed": [],
            "unavailable": ["Wednesday"],
        },
        {
            "name": "TEST 3: Multiple Scheduling Conflicts",
            "schedule": {
                "Monday": "Upper Body",
                "Tuesday": "Cardio",
                "Wednesday": "Legs",
                "Thursday": "Upper Body",
                "Friday": "Cardio",
                "Saturday": "Upper Body",
            },
            "missed": ["Wednesday", "Friday"],
            "unavailable": ["Thursday"],
        },
        {
            "name": "TEST 4: Multiple Leg Workouts",
            "schedule": {
                "Monday": "Legs",
                "Tuesday": "Cardio",
                "Wednesday": "Upper Body",
                "Thursday": "Cardio",
                "Friday": "Upper Body",
                "Saturday": "Legs",
            },
            "missed": [],
            "unavailable": [],
        },
        {
            "name": "TEST 5: No Feasible Day",
            "schedule": {
                "Monday": "Upper Body",
                "Tuesday": "Cardio",
                "Wednesday": "Upper Body",
                "Thursday": "Cardio",
                "Friday": "Upper Body",
                "Saturday": "Legs",
            },
            "missed": ["Saturday"],
            "unavailable": ["Sunday"],
        },
    ]

    for test in scenarios:
        print("\n" + "=" * 55)
        print(test["name"])
        print("=" * 55)
        print("Input schedule:")
        for day, workout in test["schedule"].items():
            print(f"  {day}: {workout}")
        print(f"Missed workouts: {test['missed']}")
        print(f"Unavailable days: {test['unavailable']}")

        try:
            workout_agent(
                test["schedule"],
                missed_workouts=test["missed"],
                unavailable_days=test["unavailable"],
            )
        except Exception as error:
            print(f"Scenario error: {error}")

        print(f"Finished: {test['name']}")

    print("\nAll five scenario runs have finished.")
