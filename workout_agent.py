def schedule_workouts(workouts, unavailable_days):
    """
    Adjust a weekly workout schedule when certain days become unavailable.
    """

    updated_schedule = workouts.copy()

    for day in unavailable_days:
        if day in updated_schedule:
            workout_to_move = updated_schedule.pop(day)

            # Find the first available day later in the week
            days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]

            current_index = days.index(day)

            for new_day in days[current_index + 1:]:
                if new_day not in updated_schedule and new_day not in unavailable_days:
                    updated_schedule[new_day] = workout_to_move
                    break

    return updated_schedule


# Example schedule
workouts = {
    "Monday": "Upper Body",
    "Wednesday": "Cardio",
    "Saturday": "Legs"
}

unavailable_days = ["Saturday"]

new_schedule = schedule_workouts(workouts, unavailable_days)

print("Original schedule:")
print(workouts)

print("\nUpdated schedule:")
print(new_schedule)
