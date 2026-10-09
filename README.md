# Workout-Scheduler-Agent
# Workout Scheduler Agent

## Overview

The Workout Scheduler Agent helps reorganize a weekly workout schedule when workouts are missed or plans change. It uses Gemini to decide what action to take and a Python scheduling tool to check and apply possible changes.

## Problem

When I miss a workout or become unavailable on a planned day, I have to figure out how to rearrange the rest of my week while respecting recovery and workout preferences. This agent helps evaluate those changes and explains when a workout cannot be rescheduled.

## How It Works

The project has two main Python files:

- `workout_agent.py`: Contains the scheduling logic that evaluates the schedule and attempts to make valid changes.
- `workout_agent_ai.py`: Implements the AI agent loop using the Gemini API. The model evaluates the situation, chooses an action, receives the scheduling tool's result, and decides whether to continue or stop.

The agent loop has a maximum of two iterations to prevent endless retries.

## Constraints

The scheduler considers the following constraints:

- Avoid scheduling leg workouts on consecutive days.
- Keep one leg-focused workout per week.
- Respect unavailable days.
- Preserve recovery time when possible.
- If no feasible day exists, explain that the missed workout cannot be rescheduled that week.

## Technology

- Python
- Google Gemini API
- Google GenAI Python SDK

## Setup

1. Install the required Python package:

   ```bash
   pip install google-genai
   ```

2. Set the `GEMINI_API_KEY` environment variable using a Gemini API key from Google AI Studio. Do not place the key directly in the source code or commit it to GitHub.

3. Run the AI agent:

   ```bash
   python workout_agent_ai.py
   ```

## Testing

The agent was tested with the following scenarios:

1. Missed workout
2. Unavailable day
3. Multiple scheduling conflicts
4. Multiple leg workouts
5. No feasible day to reschedule a missed workout

These tests helped verify the scheduling behavior and the agent's response when a workout could not be moved.

## Limitations and Future Improvements

The agent depends on the quality of the model's decisions and the scheduling tool's available options. Future improvements could include testing more complex tradeoffs, adding more workout preferences, and allowing users to enter their schedules interactively.

## Privacy and Security

Complete project README

Use synthetic workout schedules for testing. Store API keys in environment variables or a secrets manager, and never commit API keys or other sensitive information to the repository.
