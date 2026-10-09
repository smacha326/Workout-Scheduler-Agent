import json
import os

from google import genai
from google.genai import types

from workout_agent import DAYS, is_leg_workout, reschedule_workouts


MODEL = "gemini-3.5-flash-lite"
MAX_ITERATIONS = 2

CONSTRAINTS = [
    "Schedule no more than one leg-focused workout per week.",
    "Never schedule leg workouts on consecutive days.",
    "Never schedule a workout on an unavailable day.",
    "Preserve rest and recovery when possible.",
    "Cardio is flexible and may be sacrificed if necessary.",
]


def _field(value, name):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _extract_response_text(response):
    output_text = _field(response, "output_text")

    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts = []

    for step in _field(response, "steps") or []:
        if _field(step, "type") != "model_output":
            continue

        for content in _field(step, "content") or []:
            part = _field(content, "text")
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())

    return "\n".join(parts)


def _unresolved_days(tool_result, original_schedule):
    """Find source days for workouts the Python tool could not place."""
    if not isinstance(tool_result, dict):
        return []

    failure_lines = [
        change for change in tool_result.get("changes", [])
        if isinstance(change, str)
        and "Could not find an available day" in change
    ]

    return [
        day for day in DAYS
        if day in original_schedule
        and any(f"from {day} (" in line for line in failure_lines)
    ]


def _is_no_feasible_day(tool_result):
    if not isinstance(tool_result, dict):
        return False

    return any(
        isinstance(change, str)
        and "Could not find an available day" in change
        for change in tool_result.get("changes", [])
    )


def _hard_conflict(schedule, pending_missed, unavailable_days):
    """Check hard constraints independently of Gemini."""
    if pending_missed:
        return True

    for day in unavailable_days:
        if (
            day in schedule
            and str(schedule[day]).strip().lower() != "rest"
        ):
            return True

    leg_days = [
        day for day, workout in schedule.items()
        if is_leg_workout(workout)
    ]

    return len(leg_days) > 1


def _tradeoff_candidates(
    schedule, unresolved_days, original_schedule, unavailable_days
):
    """Find safe Cardio days when an unresolved leg session needs a slot."""
    unavailable = set(unavailable_days)

    unresolved_workouts = [
        original_schedule[day]
        for day in unresolved_days
        if day in original_schedule
    ]

    # This trade-off is for an unresolved leg workout, not another Cardio day.
    if not any(is_leg_workout(w) for w in unresolved_workouts):
        return []

    # Never introduce a second leg-focused workout.
    if any(is_leg_workout(w) for w in schedule.values()):
        return []

    return [
        day for day in DAYS
        if day in schedule
        and day not in unavailable
        and "cardio" in str(schedule[day]).lower()
    ]


def _request_decision(client, situation, tool_result=None):
    """Ask Gemini to choose an action from the current situation."""
    if tool_result is None:
        if situation.get("missed_workouts"):
            rule = (
                "A workout was missed. Choose action='reschedule' so the "
                "Python tool can check for an empty available day."
            )
        else:
            rule = (
                "Review the schedule. Choose 'reschedule' if a hard "
                "constraint is violated; otherwise choose 'keep'."
            )

        prompt = (
            "You are the reasoning layer of a workout scheduling agent. "
            "Do not edit the schedule yourself; Python executes changes.\n"
            f"{rule}\n"
            "Return JSON with exactly two fields: action and reason.\n"
            f"Situation: {json.dumps(situation)}"
        )

    elif _is_no_feasible_day(tool_result):
        candidates = situation.get(
            "flexible_replacement_candidates", []
        )

        prompt = (
            "The Python scheduling tool could not place a workout on an "
            "empty available day. Make a trade-off decision.\n"
            "Hard constraints are non-negotiable: never schedule on an "
            "unavailable day and never create more than one leg-focused "
            "workout in the week.\n"
            f"Safe Cardio replacement days: {json.dumps(candidates)}\n"
            "Choose 'replace_flexible' only if a safe candidate exists. "
            "Otherwise choose 'drop'. Do not choose 'keep'. Python will "
            "validate and execute the decision.\n"
            "Return JSON with exactly two fields: action and reason.\n"
            f"Situation: {json.dumps(situation)}\n"
            f"Previous Python tool result: {json.dumps(tool_result)}"
        )

    else:
        prompt = (
            "Review the schedule produced by the Python tool. The tool "
            "successfully resolved the scheduling conflicts. Choose 'keep' "
            "and briefly explain why no further change is needed. Do not "
            "drop a workout that was successfully rescheduled.\n"
            "Return JSON with exactly two fields: action and reason.\n"
            f"Situation: {json.dumps(situation)}\n"
            f"Previous Python tool result: {json.dumps(tool_result)}"
        )

    try:
        response = client.interactions.create(
            model=MODEL,
            input=prompt,
            generation_config={"thinking_level": "minimal"},
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "reschedule",
                                "keep",
                                "replace_flexible",
                                "drop",
                            ],
                        },
                        "reason": {"type": "string"},
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

    allowed = {"reschedule", "keep", "replace_flexible", "drop"}

    if not isinstance(result, dict) or result.get("action") not in allowed:
        raise ValueError(
            f"Gemini response must use one of: {sorted(allowed)}"
        )

    if (
        not isinstance(result.get("reason"), str)
        or not result["reason"].strip()
    ):
        raise ValueError("Gemini response must include a non-empty reason.")

    return {
        "action": result["action"],
        "reason": result["reason"].strip(),
    }


def _apply_flexible_tradeoff(
    current_schedule,
    original_schedule,
    unresolved_days,
    unavailable_days,
):
    """Replace an available Cardio session with a missed leg workout safely."""
    updated = current_schedule.copy()
    unavailable = set(unavailable_days)

    missed_leg = next(
        (
            original_schedule[day]
            for day in unresolved_days
            if day in original_schedule
            and is_leg_workout(original_schedule[day])
        ),
        None,
    )

    if missed_leg is None:
        return (
            updated,
            ["No unresolved leg workout was available for trade-off."],
            False,
        )

    if any(is_leg_workout(workout) for workout in updated.values()):
        return (
            updated,
            ["Trade-off rejected: a leg workout is already scheduled."],
            False,
        )

    target_day = next(
        (
            day for day in DAYS
            if day in updated
            and day not in unavailable
            and "cardio" in str(updated[day]).lower()
        ),
        None,
    )

    if target_day is None:
        return updated, ["No safe Cardio replacement was available."], False

    updated[target_day] = missed_leg

    # Validate hard constraints after the change.
    if (
        target_day in unavailable
        or sum(
            1 for workout in updated.values()
            if is_leg_workout(workout)
        ) > 1
    ):
        return (
            current_schedule.copy(),
            ["Trade-off rejected by constraint validation."],
            False,
        )

    changes = [
        f"Trade-off: replaced Cardio on {target_day} with {missed_leg}; "
        "the missed leg workout was prioritized over Cardio."
    ]

    return updated, changes, True


def workout_agent(schedule, missed_workouts=None, unavailable_days=None):
    missed_workouts = list(missed_workouts or [])
    unavailable_days = list(unavailable_days or [])

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your environment before running."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=15000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    original_schedule = dict(schedule)
    updated_schedule = dict(schedule)
    pending_missed = list(missed_workouts)
    tool_result = None
    changes = []

    final_response = (
        "Iteration limit reached before a final decision was made."
    )

    for iteration in range(MAX_ITERATIONS):
        situation = {
            "planned_schedule": updated_schedule,
            "missed_workouts": pending_missed if tool_result is None else [],
            "original_missed_workouts": missed_workouts,
            "unavailable_days": unavailable_days,
            "constraints": CONSTRAINTS,
        }

        unresolved_days = []
        candidates = []

        if (
            tool_result is not None
            and _is_no_feasible_day(tool_result)
        ):
            unresolved_days = _unresolved_days(
                tool_result, original_schedule
            )

            candidates = _tradeoff_candidates(
                updated_schedule,
                unresolved_days,
                original_schedule,
                unavailable_days,
            )

            situation["unresolved_workout_days"] = unresolved_days
            situation["flexible_replacement_candidates"] = candidates

        result = _request_decision(client, situation, tool_result)
        action = result["action"]
        reason = result["reason"]

        # Python independently validates Gemini's proposed action.
        if tool_result is None:
            if _hard_conflict(
                updated_schedule, pending_missed, unavailable_days
            ):
                action = "reschedule"
            elif action not in {"keep", "reschedule"}:
                action = "keep"

        elif _is_no_feasible_day(tool_result):
            if candidates:
                # A safe trade-off exists, so do not accept an incorrect
                # model decision to drop the missed leg workout.
                action = "replace_flexible"
                reason = (
                    f"No empty day is available, but Cardio can safely be "
                    f"replaced on {', '.join(candidates)}. Replacing one "
                    "Cardio session with Legs preserves the one-leg-workout "
                    "limit and avoids unavailable days."
                )
            else:
                action = "drop"
                reason = (
                    "No safe empty day or flexible-Cardio replacement is "
                    "available, so the unresolved workout must be dropped "
                    "for this week."
                )

        else:
            # Never let Gemini undo a successful Python scheduling result.
            action = "keep"

        print(f"Agent decision (iteration {iteration + 1}): {action}")
        print(f"Agent reason: {reason}")

        if action == "keep":
            final_response = reason
            print("Tool called: no")
            break

        if action == "drop":
            unresolved_days = (
                _unresolved_days(tool_result, original_schedule)
                if tool_result is not None
                else []
            )

            prior_successes = [
                change
                for change in (tool_result or {}).get("changes", [])
                if "Could not find an available day" not in change
            ]

            drop_changes = [
                f"Dropped {original_schedule[day]} from {day} for this week "
                "because no safe scheduling option was available."
                for day in unresolved_days
                if day in original_schedule
            ]

            changes = prior_successes + drop_changes
            final_response = reason
            print("Agent decision: drop unresolved workout(s) for this week.")
            break

        if action == "replace_flexible":
            updated_schedule, tradeoff_changes, success = (
                _apply_flexible_tradeoff(
                    updated_schedule,
                    original_schedule,
                    unresolved_days,
                    unavailable_days,
                )
            )

            prior_successes = [
                change
                for change in (tool_result or {}).get("changes", [])
                if "Could not find an available day" not in change
            ]

            # Keep successful earlier changes, but remove stale failure messages.
            changes = prior_successes + tradeoff_changes

            print("Tool called: apply_flexible_tradeoff")
            print("Tool result:")

            for change in changes:
                print(f"- {change}")

            final_response = (
                reason
                if success
                else (
                    "The trade-off could not be applied safely; the "
                    "unresolved workout was left off this week's schedule."
                )
            )
            break

        # Execute the deterministic Python scheduling tool.
        print("Tool called: reschedule_workouts")

        updated_schedule, changes = reschedule_workouts(
            updated_schedule,
            missed_workouts=pending_missed,
            unavailable_days=unavailable_days,
        )

        tool_result = {
            "updated_schedule": updated_schedule.copy(),
            "changes": changes.copy(),
        }

        print("Tool result:")
        for change in changes:
            print(f"- {change}")

        pending_missed = []

        if _is_no_feasible_day(tool_result):
            if iteration < MAX_ITERATIONS - 1:
                # Let Gemini assess the failed attempt and make a trade-off.
                continue

            final_response = (
                "No feasible empty day was found within the iteration limit."
            )
            break

        if iteration < MAX_ITERATIONS - 1:
            # Feed the successful result back to Gemini for review.
            continue

        final_response = (
            "The Python tool updated the schedule; "
            "the iteration limit was reached."
        )

    print(f"Final agent response: {final_response}")
    print("Final updated schedule:")

    for day in DAYS:
        if day in updated_schedule:
            print(f"{day}: {updated_schedule[day]}")

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
            "name": "TEST 5: No Feasible Day / Trade-off",
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