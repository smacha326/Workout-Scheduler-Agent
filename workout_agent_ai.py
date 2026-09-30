import json
import os
from contextlib import redirect_stdout
from io import StringIO

from google import genai


# The scheduler module has a sample run at import time; keep this demo focused.
with redirect_stdout(StringIO()):
    from workout_agent import reschedule_workouts


MODEL = "gemini-3.8-flash"
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


def workout_agent(schedule, missed_workouts=None, unavailable_days=None):
    missed_workouts = missed_workouts or []
    unavailable_days = unavailable_days or []
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable to run the agent.")

    situation = {
        "schedule": schedule,
        "missed_workouts": missed_workouts,
        "unavailable_days": unavailable_days,
        "constraints": CONSTRAINTS,
    }
    prompt = (
        "You are the reasoning layer of a workout scheduling agent. Decide whether "
        "the schedule needs adjustment. Do not create or rewrite a schedule; a separate "
        "verified tool handles any changes. Return only a JSON object with exactly these "
        'fields: {"action":"reschedule" or "keep","reason":"brief explanation"}.\n\n'
        f"Situation: {json.dumps(situation)}"
    )

    client = genai.Client(api_key=api_key)
    response = client.interactions.create(model=MODEL, input=prompt)
    response_text = _extract_response_text(response)
    try:
        result = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as error:
        start = response_text.find("{")
        if start >= 0:
            try:
                result, _ = json.JSONDecoder().raw_decode(response_text[start:])
            except json.JSONDecodeError:
                raise ValueError("Gemini did not return valid JSON for its decision.") from error
        else:
            raise ValueError("Gemini did not return valid JSON for its decision.") from error

    if not isinstance(result, dict) or result.get("action") not in {"reschedule", "keep"}:
        raise ValueError('Gemini JSON must contain action "reschedule" or "keep".')
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        raise ValueError("Gemini JSON must include a non-empty reason.")

    action = result["action"]
    reason = result["reason"].strip()
    print(f"Agent decision: {action}")
    print(f"Agent reason: {reason}")

    if action == "reschedule":
        print("Tool called: reschedule_workouts")
        updated_schedule, changes = reschedule_workouts(
            schedule,
            missed_workouts=missed_workouts,
            unavailable_days=unavailable_days,
        )
        print("Tool result:")
        if changes:
            for change in changes:
                print(f"- {change}")
        else:
            print("- The scheduler made no changes.")
    else:
        updated_schedule, changes = schedule.copy(), []
        print("Tool called: no")
        print("Tool result: not applicable; the agent kept the schedule unchanged.")

    print("Final updated schedule:")
    for day, workout in updated_schedule.items():
        print(f"{day}: {workout}")
    return updated_schedule, changes


if __name__ == "__main__":

    schedule = {
        "Monday": "Upper Body",
        "Wednesday": "Cardio",
        "Saturday": "Legs"
    }

    workout_agent(
        schedule,
        missed_workouts=["Saturday"]
    )