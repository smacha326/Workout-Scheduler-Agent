DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def is_leg_workout(workout):
    """Return True for leg-, lower-body-, or glute-focused workouts."""
    text = str(workout).strip().lower()
    return "leg" in text or "lower body" in text or "glute" in text


def _is_rest(workout):
    return str(workout).strip().lower() == "rest"


def reschedule_workouts(schedule, missed_workouts=None, unavailable_days=None):
    """Apply scheduling rules and report workouts that could not be placed.

    The tool makes deterministic schedule updates. The AI agent interprets the
    returned results and chooses what to do about any unresolved trade-off.

    Returns (updated_schedule, change_messages).
    """
    missed_workouts = list(dict.fromkeys(missed_workouts or []))
    unavailable_days = set(unavailable_days or [])

    invalid_days = (
        set(schedule) | set(missed_workouts) | unavailable_days
    ) - set(DAYS)

    if invalid_days:
        raise ValueError(f"Unknown day name(s): {sorted(invalid_days)}")

    updated = dict(schedule)
    changes = []
    pending = []

    # Remove all missed sessions first, preventing workouts from being
    # overwritten when multiple sessions must be rescheduled.
    for day in missed_workouts:
        workout = updated.pop(day, None)
        if workout is not None and not _is_rest(workout):
            pending.append((day, workout, "missed"))

    # Remove unavailable-day sessions and queue them for rescheduling.
    for day in DAYS:
        if day in unavailable_days and day in updated:
            workout = updated.pop(day)
            if not _is_rest(workout):
                pending.append((day, workout, "unavailable"))
                changes.append(
                    f"Removed {workout} from unavailable day {day}."
                )

    # If the input contains multiple leg workouts, preserve the first in
    # calendar order and report the extra session removed by the tool.
    existing_leg_days = [
        day for day in DAYS
        if day in updated and is_leg_workout(updated[day])
    ]

    for day in existing_leg_days[1:]:
        workout = updated.pop(day)
        changes.append(
            f"Removed extra leg workout from {day} ({workout}); "
            "only one leg-focused workout is allowed per week."
        )

    # Try leg sessions first because the weekly leg-workout constraint
    # is more restrictive than the flexible-workout constraint.
    pending.sort(key=lambda item: 0 if is_leg_workout(item[1]) else 1)

    for original_day, workout, reason in pending:
        # Do not add a second leg session alongside an existing one.
        # Report the unresolved case so the AI can make the trade-off decision.
        if is_leg_workout(workout) and any(
            is_leg_workout(existing) for existing in updated.values()
        ):
            changes.append(
                f"Could not find an available day to reschedule {workout} "
                f"from {original_day} ({reason}): another leg workout remains."
            )
            continue

        # Prefer days after the original day, then wrap to earlier days.
        origin_index = DAYS.index(original_day)
        candidates = DAYS[origin_index + 1:] + DAYS[:origin_index]

        target_day = next(
            (
                day for day in candidates
                if day not in updated and day not in unavailable_days
            ),
            None,
        )

        if target_day is None:
            changes.append(
                f"Could not find an available day to reschedule {workout} "
                f"from {original_day} ({reason})."
            )
            continue

        updated[target_day] = workout
        changes.append(
            f"Moved {workout} from {original_day} to {target_day} "
            f"after it was {reason}."
        )

    # Validate hard constraints before returning the tool result.
    for day in unavailable_days:
        if day in updated and not _is_rest(updated[day]):
            raise RuntimeError(
                f"Scheduling error: {day} is unavailable but has "
                f"{updated[day]} scheduled."
            )

    final_leg_days = [
        day for day in DAYS
        if day in updated and is_leg_workout(updated[day])
    ]

    if len(final_leg_days) > 1:
        raise RuntimeError(
            f"Scheduling error: multiple leg workouts remain: {final_leg_days}"
        )

    if not changes:
        changes.append("No schedule changes were needed.")

    return updated, changes