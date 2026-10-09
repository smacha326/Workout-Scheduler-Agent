# Workout Scheduler Agent

## Overview

The Workout Scheduler Agent helps reorganize a weekly workout schedule when workouts are missed or plans change. It uses Google's Gemini API to reason about scheduling conflicts and choose an action, while a Python scheduling tool checks and applies schedule changes.

## Problem

When I miss a workout or become unavailable on a planned day, I have to figure out how to rearrange the rest of my week while respecting recovery and workout preferences. This agent helps evaluate the remaining schedule, resolve conflicts, and make trade-offs when a workout cannot be moved to an empty day.

## How It Works

The project has two main Python files:

- `workout_agent.py`: Contains the Python scheduling tool. It removes missed or unavailable sessions, attempts to move workouts to available days, and reports unresolved scheduling conflicts.
- `workout_agent_ai.py`: Implements the AI agent loop using the Gemini API. Gemini evaluates the current situation, chooses an action, receives the scheduling tool's result, and decides what to do next. Python validates hard constraints and executes the selected action.

The agent uses a bounded loop with a maximum of three iterations to prevent endless retries and limit API calls.

## Constraints

The scheduler considers the following constraints:

- Schedule no more than one leg-focused workout per week.
- Avoid consecutive leg workouts.
- Never schedule workouts on unavailable days.
- Preserve rest and recovery when possible.
- Treat Cardio as a flexible workout that can be sacrificed when appropriate.
- When no empty day is available, evaluate whether replacing a flexible workout or dropping the missed workout is the better trade-off.

## Technology

- Python
- Google Gemini API
- Google GenAI Python SDK

## Setup

### 1. Install the required package

```bash
pip install google-genai
```

### 2. Configure the API key

Set the `GEMINI_API_KEY` environment variable using a valid Gemini API key from Google AI Studio.

For a Bash terminal, you can set it for the current terminal session with:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

Replace the placeholder with your own key. Never place the real key in the source code or commit it to GitHub.

### 3. Run the agent

```bash
python workout_agent_ai.py
```

The program runs five example scheduling scenarios and prints the agent's decisions, tool results, and final schedules.

## Testing

The agent was tested with five scenarios:

1. **Missed Workout:** Attempts to move a missed workout to an available day.
2. **Unavailable Day:** Moves a scheduled workout off a day when exercise is unavailable.
3. **Multiple Scheduling Conflicts:** Processes multiple missed workouts and an unavailable day.
4. **Multiple Leg Workouts:** Removes an extra leg-focused workout to maintain the weekly limit.
5. **No Feasible Day / Trade-off:** Evaluates replacing a flexible Cardio session with a missed leg workout when no empty day is available.

The scenarios help verify scheduling behavior, hard-constraint validation, and the agent's response to unresolved conflicts.

## Limitations and Future Improvements

The agent currently uses predefined example schedules and a limited set of workout constraints. Its scheduling tool uses deterministic placement rules, and the model's decisions are limited by the available actions and the iteration cap. Future improvements could include interactive schedule input, more complex recovery preferences, additional test scenarios, and more detailed evaluation of trade-offs.

## Privacy and Security

Use synthetic workout schedules or other non-sensitive data for testing. Store API keys in environment variables or a secrets manager, and never commit API keys or other sensitive information to the repository.