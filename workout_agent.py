DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


def reschedule_workouts(schedule, missed_workouts=None, unavailable_days=None):
    """
    Adjust a weekly workout schedule when workouts are missed
    or days become unavailable.
    """

    missed_workouts = missed_workouts or []
    unavailable_days = unavailable_days or []

    updated_schedule = schedule.copy()
    changes = []

    # Keep track of days that became empty because a missed
    # workout could not be rescheduled.
    protected_empty_days = set()

    # 1. Reschedule workouts that were missed.
    for day, workout in list(updated_schedule.items()):
        if day in missed_workouts:
            del updated_schedule[day]

            moved = False
            current_index = DAYS.index(day)

            # First, look for an available day later in the week.
            candidate_days = DAYS[current_index + 1:]

            # If needed, also consider available days earlier in the week.
            candidate_days += DAYS[:current_index]

            for new_day in candidate_days:
                if (
                    new_day not in updated_schedule
                    and new_day not in unavailable_days
                ):
                    updated_schedule[new_day] = workout

                    changes.append(
                        f"Moved {workout} from {day} to {new_day} "
                        "after it was missed."
                    )

                    moved = True
                    break

            if not moved:
                protected_empty_days.add(day)

                changes.append(
                    f"Could not find an available day to reschedule "
                    f"{workout} from {day}."
                )

    # 2. Move workouts from unavailable days.
    for day in unavailable_days:
        if (
            day in updated_schedule
            and day not in protected_empty_days
        ):
            workout = updated_schedule[day]

            # Look for an available day later in the week.
            for new_day in DAYS[DAYS.index(day) + 1:]:
                if (
                    new_day not in updated_schedule
                    and new_day not in unavailable_days
                    and new_day not in protected_empty_days
                ):
                    updated_schedule[new_day] = workout
                    del updated_schedule[day]

                    changes.append(
                        f"Moved {workout} from {day} to {new_day}."
                    )

                    break

    # 3. Protect the leg-day rule.
    leg_days = [
        day for day, workout in updated_schedule.items()
        if "leg" in workout.lower()
    ]

    if len(leg_days) > 1:
        # Keep the first leg day and remove extra leg workouts.
        for day in leg_days[1:]:
            workout = updated_schedule.pop(day)

            changes.append(
                f"Removed extra leg workout from {day} "
                "to maintain one leg-focused workout per week."
            )

    # 4. Check for consecutive leg days.
    sorted_days = sorted(
        updated_schedule.keys(),
        key=lambda day: DAYS.index(day)
    )

    for i in range(len(sorted_days) - 1):
        day1 = sorted_days[i]
        day2 = sorted_days[i + 1]

        workout1 = updated_schedule[day1].lower()
        workout2 = updated_schedule[day2].lower()

        if "leg" in workout1 and "leg" in workout2:
            changes.append(
                f"Warning: {day1} and {day2} are consecutive leg days. "
                "Recovery may need to be adjusted."
            )

    return updated_schedule, changes


# Test Scenario 6: No feasible day

schedule = {
    "Monday": "Upper Body",
    "Tuesday": "Cardio",
    "Wednesday": "Legs"
}

new_schedule, changes = reschedule_workouts(
    schedule,
    missed_workouts=["Wednesday"],
    unavailable_days=[
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday"
    ]
)

print("SCENARIO 6: NO FEASIBLE DAY")

print("\nOriginal schedule:")
print(schedule)

print("\nUpdated schedule:")
print(new_schedule)

print("\nDecisions and changes:")
for change in changes:
    print("-", change)