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
    """Move workouts to empty available days and report unresolved conflicts."""
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

    # Remove all missed workouts first to prevent overwriting sessions.
    for day in missed_workouts:
        workout = updated.pop(day, None)
        if workout is not None and not _is_rest(workout):
            pending.append((day, workout, "missed"))

    # Remove workouts from unavailable days and queue them for rescheduling.
    for day in DAYS:
        if day in unavailable_days and day in updated:
            workout = updated.pop(day)
            if not _is_rest(workout):
                pending.append((day, workout, "unavailable"))
                changes.append(
                    f"Removed {workout} from unavailable day {day}."
                )

    # Keep at most one existing leg-focused workout.
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

    # Try leg-focused sessions first because they have the strictest constraint.
    pending.sort(key=lambda item: 0 if is_leg_workout(item[1]) else 1)

    for original_day, workout, reason in pending:
        if is_leg_workout(workout) and any(
            is_leg_workout(existing) for existing in updated.values()
        ):
            changes.append(
                f"Dropped {workout} from {original_day}: another "
                "leg-focused workout is already scheduled this week."
            )
            continue

        origin_index = DAYS.index(original_day)
        candidate_days = DAYS[origin_index + 1:] + DAYS[:origin_index]

        target_day = next(
            (
                day for day in candidate_days
                if day not in updated and day not in unavailable_days
            ),
            None,
        )

        if target_day is None:
            changes.append(
                f"Could not find an available day to reschedule "
                f"{workout} from {original_day} ({reason})."
            )
            continue

        updated[target_day] = workout
        changes.append(
            f"Moved {workout} from {original_day} to {target_day} "
            f"after it was {reason}."
        )

    # Validate hard constraints before returning the schedule.
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