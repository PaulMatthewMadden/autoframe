# New Features

All recommendations (Recs) for new features are prefixed with a 'F'.

## Recs

### F1 : Camera Auto Select
- [x] Started
- [x] Completed

If the system detects only 1 camera, it should automatically select it without prompting the user.

## Completed

### F1 : Camera Auto Select

Date Completed: 2026-04-29

#### Results

Implemented auto-selection logic in `main.py`:

- Added conditional check for `len(device_names) == 1`
- When only 1 camera detected, automatically selects index 0 and prints info message
- When multiple cameras detected, shows the existing selection menu
- Eliminates unnecessary user interaction for single-camera setups
