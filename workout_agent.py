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

    # 1. Remove workouts that were missed.
    for day, workout in list(updated_schedule.items()):
        if day in missed_workouts:
            del updated_schedule[day]
            changes.append(
                f"{workout} on {day} was missed and needs to be rescheduled."
            )

    # 2. Move workouts from unavailable days.
    for day in unavailable_days:
        if day in updated_schedule:
            workout = updated_schedule.pop(day)

            moved = False

            # Look for an available day later in the week.
            for new_day in DAYS[DAYS.index(day) + 1:]:
                if (
                    new_day not in updated_schedule
                    and new_day not in unavailable_days
                ):
                    updated_schedule[new_day] = workout
                    changes.append(
                        f"Moved {workout} from {day} to {new_day}."
                    )
                    moved = True
                    break

            if not moved:
                changes.append(
                    f"Could not find an available day for {workout} from {day}."
                )

    # 3. Protect the leg-day rule.
    leg_days = [
        day for day, workout in updated_schedule.items()
        if "leg" in workout.lower()
    ]

    if len(leg_days) > 1:
        # Keep the first leg day and move the additional one.
        for day in leg_days[1:]:
            workout = updated_schedule.pop(day)

            for new_day in DAYS:
                if (
                    new_day not in updated_schedule
                    and new_day not in unavailable_days
                ):
                    updated_schedule[new_day] = workout
                    changes.append(
                        f"Moved extra leg workout from {day} to {new_day} "
                        "to keep one leg-focused workout per week."
                    )
                    break

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


# Example weekly schedule
schedule = {
    "Monday": "Upper Body",
    "Wednesday": "Cardio",
    "Saturday": "Legs"
}

# Example: Saturday becomes unavailable
unavailable_days = ["Saturday"]

# No workouts were missed in this example
missed_workouts = []

new_schedule, changes = reschedule_workouts(
    schedule,
    missed_workouts,
    unavailable_days
)

print("ORIGINAL SCHEDULE")
print(schedule)

print("\nUPDATED SCHEDULE")
print(new_schedule)

print("\nDECISIONS AND CHANGES")
for change in changes:
    print("-", change)
