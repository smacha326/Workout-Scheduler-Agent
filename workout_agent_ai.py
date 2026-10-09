import json
import os

from google import genai
from google.genai import types

from workout_agent import DAYS, is_leg_workout, reschedule_workouts


MODEL = "gemini-3.5-flash-lite"

# A fixed cap prevents an unbounded loop of paid API calls.
MAX_ITERATIONS = 3

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
    """Extract text from the Gemini SDK response."""
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
    """Identify the source days of workouts the tool could not place."""
    if not isinstance(tool_result, dict):
        return []

    failure_lines = [
        change
        for change in tool_result.get("changes", [])
        if isinstance(change, str)
        and "Could not find an available day to reschedule" in change
    ]

    return [
        day
        for day in DAYS
        if day in original_schedule
        and any(f"from {day} (" in line for line in failure_lines)
    ]


def _is_no_feasible_day(tool_result):
    """Return True when the latest tool execution left a placement unresolved."""
    if not isinstance(tool_result, dict):
        return False

    return any(
        isinstance(change, str)
        and "Could not find an available day to reschedule" in change
        for change in tool_result.get("changes", [])
    )


def _hard_conflicts(schedule, pending_missed, unavailable_days):
    """Report objective constraint violations for Python validation."""
    conflicts = []

    if pending_missed:
        conflicts.append(
            f"Missed sessions still need processing: {pending_missed}"
        )

    for day in unavailable_days:
        if (
            day in schedule
            and str(schedule[day]).strip().lower() != "rest"
        ):
            conflicts.append(
                f"{day} is unavailable but has {schedule[day]} scheduled"
            )

    leg_days = [
        day
        for day, workout in schedule.items()
        if is_leg_workout(workout)
    ]

    if len(leg_days) > 1:
        conflicts.append(
            f"Multiple leg-focused workouts remain on {leg_days}"
        )

    return conflicts


def _tradeoff_candidates(
    schedule, unresolved_days, original_schedule, unavailable_days
):
    """Find Cardio sessions that can safely be sacrificed for unresolved work."""
    unavailable = set(unavailable_days)

    unresolved_workouts = [
        original_schedule[day]
        for day in unresolved_days
        if day in original_schedule
    ]

    # This trade-off is intended to place a missed leg session.
    if not any(is_leg_workout(workout) for workout in unresolved_workouts):
        return []

    # Never create a second leg-focused workout.
    if any(is_leg_workout(workout) for workout in schedule.values()):
        return []

    return [
        day
        for day in DAYS
        if day in schedule
        and day not in unavailable
        and "cardio" in str(schedule[day]).lower()
    ]


def _request_decision(client, situation, tool_result=None):
    """Ask Gemini to choose the next action from the latest observations."""
    unresolved = _is_no_feasible_day(tool_result)

    if tool_result is None:
        prompt = (
            "You are the reasoning layer of a workout scheduling agent.\n"
            "Inspect the current schedule, missed sessions, unavailable days, "
            "and hard constraints. Choose the next action.\n"
            "Use 'reschedule' when a missed session or hard conflict needs "
            "the Python scheduling tool. Use 'keep' only when the schedule "
            "is already acceptable.\n"
            "Do not edit the schedule yourself; Python executes actions.\n"
            "Return JSON with exactly two fields: action and reason.\n"
            f"Situation: {json.dumps(situation)}"
        )

    elif unresolved:
        candidates = situation.get(
            "flexible_replacement_candidates", []
        )

        prompt = (
            "The Python scheduling tool tried to move a workout and reported "
            "that it could not find a safe empty day. Make the trade-off "
            "decision based on this result.\n"
            "Hard constraints are non-negotiable: never use an unavailable "
            "day and never create more than one leg-focused workout in the week.\n"
            f"Safe Cardio replacement candidates: {json.dumps(candidates)}\n"
            "Choose 'replace_flexible' to sacrifice one of these Cardio "
            "sessions if that is the best trade-off, or 'drop' if you judge "
            "that dropping the missed workout is preferable. If there are no "
            "safe candidates, choose 'drop'. Do not choose 'keep' or "
            "'reschedule' for this unresolved trade-off.\n"
            "The interpretation and trade-off choice are yours. Python will "
            "validate and execute the chosen action. Explain your reasoning.\n"
            "Return JSON with exactly two fields: action and reason.\n"
            f"Situation: {json.dumps(situation)}\n"
            f"Previous Python tool result: {json.dumps(tool_result)}"
        )

    else:
        conflicts = situation.get("hard_conflicts", [])

        prompt = (
            "Review the actual result returned by the Python scheduling tool.\n"
            f"Remaining hard conflicts reported by validation: "
            f"{json.dumps(conflicts)}\n"
            "Choose 'keep' if the schedule is acceptable. Choose 'reschedule' "
            "only if another tool action is needed to resolve a remaining "
            "conflict. Do not undo a successfully rescheduled workout.\n"
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

    allowed = {
        "reschedule",
        "keep",
        "replace_flexible",
        "drop",
    }

    if not isinstance(result, dict) or result.get("action") not in allowed:
        raise ValueError(
            f"Gemini response must use one of: {sorted(allowed)}"
        )

    if (
        not isinstance(result.get("reason"), str)
        or not result["reason"].strip()
    ):
        raise ValueError(
            "Gemini response must include a non-empty reason."
        )

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
    """Execute an AI-chosen replacement only if it passes hard constraints."""
    updated = current_schedule.copy()
    unavailable = set(unavailable_days)

    missed_leg_day = next(
        (
            day
            for day in unresolved_days
            if day in original_schedule
            and is_leg_workout(original_schedule[day])
        ),
        None,
    )

    missed_leg = (
        original_schedule.get(missed_leg_day)
        if missed_leg_day else None
    )

    if missed_leg is None:
        return (
            updated,
            ["Trade-off rejected: no unresolved leg workout exists."],
            False,
        )

    if any(is_leg_workout(workout) for workout in updated.values()):
        return (
            updated,
            ["Trade-off rejected: another leg workout is already scheduled."],
            False,
        )

    target_day = next(
        (
            day
            for day in DAYS
            if day in updated
            and day not in unavailable
            and "cardio" in str(updated[day]).lower()
        ),
        None,
    )

    if target_day is None:
        return (
            updated,
            ["Trade-off rejected: no safe Cardio replacement exists."],
            False,
        )

    updated[target_day] = missed_leg

    # Validate the schedule after applying the proposed action.
    if (
        target_day in unavailable
        or sum(
            1 for workout in updated.values()
            if is_leg_workout(workout)
        ) > 1
    ):
        return (
            current_schedule.copy(),
            ["Trade-off rejected by hard-constraint validation."],
            False,
        )

    changes = [
        f"Trade-off applied: replaced Cardio on {target_day} with {missed_leg}."
    ]

    # If there are other unresolved workouts, report them honestly rather
    # than silently losing them after making the single Cardio replacement.
    for day in unresolved_days:
        if day != missed_leg_day and day in original_schedule:
            changes.append(
                f"Dropped unresolved {original_schedule[day]} from {day} for "
                "this week because no safe empty day remained."
            )

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
        conflicts = _hard_conflicts(
            updated_schedule,
            pending_missed if tool_result is None else [],
            unavailable_days,
        )

        situation = {
            "planned_schedule": updated_schedule,
            "missed_workouts": (
                pending_missed if tool_result is None else []
            ),
            "original_missed_workouts": missed_workouts,
            "unavailable_days": unavailable_days,
            "constraints": CONSTRAINTS,
            "hard_conflicts": conflicts,
        }

        unresolved_days = []
        candidates = []

        if tool_result is not None and _is_no_feasible_day(tool_result):
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

        # The model observes the state and chooses the next action.
        result = _request_decision(client, situation, tool_result)
        action = result["action"]
        reason = result["reason"]

        # Safety validation: Python may reject an unsafe or state-inappropriate
        # action. It does not choose between valid replace/drop trade-offs.
        if tool_result is None and conflicts:
            if action != "reschedule":
                print(
                    f"Validation rejected action '{action}': an initial hard "
                    "conflict requires the scheduling tool."
                )
                action = "reschedule"
                reason = (
                    "The schedule has a missed session or hard conflict "
                    "that must be processed."
                )

        elif _is_no_feasible_day(tool_result):
            if action == "replace_flexible" and not candidates:
                print(
                    "Validation rejected replace_flexible: "
                    "no safe candidate exists."
                )
                action = "drop"
                reason = (
                    "No safe replacement exists, so the missed workout "
                    "must be dropped this week."
                )

            elif action not in {"replace_flexible", "drop"}:
                # Fallback only when the model returns an action that is
                # invalid for this state, not when it chooses a valid trade-off.
                print(
                    f"Validation rejected action '{action}' for the "
                    "unresolved trade-off."
                )

                action = "replace_flexible" if candidates else "drop"

                reason = (
                    "Fallback after an invalid trade-off action: applying "
                    "the safe replacement."
                    if candidates
                    else
                    "Fallback after an invalid trade-off action: "
                    "no safe replacement exists."
                )

        elif tool_result is not None:
            # Do not let the model drop or replace a workout after the tool
            # successfully resolved the conflict.
            if action in {"replace_flexible", "drop"}:
                print(
                    f"Validation rejected action '{action}': the tool result "
                    "contains no unresolved placement failure."
                )
                action = "keep"
                reason = (
                    "The scheduling tool resolved the conflicts; "
                    "no workout needs to be dropped."
                )

            elif action == "keep" and conflicts:
                print("Validation rejected keep: a hard conflict remains.")
                action = "reschedule"
                reason = (
                    "A hard conflict remains, so the scheduling tool "
                    "must run again."
                )

        print(f"Agent decision (iteration {iteration + 1}): {action}")
        print(f"Agent reason: {reason}")

        if action == "keep":
            final_response = reason
            print("Tool called: no")
            break

        if action == "drop":
            unresolved_days = _unresolved_days(
                tool_result, original_schedule
            )

            previous_successes = [
                change
                for change in (tool_result or {}).get("changes", [])
                if (
                    "Could not find an available day to reschedule"
                    not in change
                )
            ]

            dropped = [
                f"Dropped {original_schedule[day]} from {day} for this week; "
                "no safe scheduling option was selected."
                for day in unresolved_days
                if day in original_schedule
            ]

            changes = previous_successes + dropped
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

            previous_successes = [
                change
                for change in (tool_result or {}).get("changes", [])
                if (
                    "Could not find an available day to reschedule"
                    not in change
                )
            ]

            changes = previous_successes + tradeoff_changes

            print("Tool called: apply_flexible_tradeoff")
            print("Tool result:")

            for change in changes:
                print(f"- {change}")

            final_response = (
                reason
                if success
                else tradeoff_changes[0]
            )
            break

        # Execute the tool that Gemini chose to call.
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
                # Feed the failed attempt back to Gemini for its judgment.
                continue

            final_response = (
                "Iteration limit reached with an unresolved scheduling conflict."
            )
            break

        if iteration < MAX_ITERATIONS - 1:
            # Feed the tool result back to Gemini; the model decides whether
            # another action is needed or whether the task is finished.
            continue

        final_response = (
            "Iteration limit reached after the scheduling tool ran."
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