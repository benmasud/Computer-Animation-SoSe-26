# Python Motion Capture Assignments

This repository contains a Python implementation of motion capture data processing for the course assignment.

## Structure

- `assignment1_filtering.py` - starter script for Assignment 1.
- `assignment2_constraints.py` - starter script for Assignment 2.
- `demo.py` - minimal viewer demo.
- `data/` - sample ASF/AMC motion capture files.
- `motion_player/` - ASF/AMC loader and 3D motion viewer.
- `optimization/` - helper functions for constrained motion editing.
- `requirements.txt` - Python dependencies required for the project.

## Setup

1. Create or activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting started

Load skeleton and motion data in Python:

```python
from motion_player.amc_parser import parse_asf, parse_amc

joints = parse_asf('data/HDM_bk.asf')
motions = parse_amc('data/HDM_bk_walk.amc')
```
## Viewer

Visualize a motion sequence using the viewer:

```python
from motion_player.Viewer3D import Viewer

viewer = Viewer(joints=(joints,), motions=(motions,), legends=('Original',))
viewer.run()
```

For assignment 1, use the optional `legend_groups` parameter to display Euler,
original, and quaternion motions in separate panels:

```python
viewer = Viewer(
    joints=(joints_euler_32, joints_euler_64, ...,
            joints_original,
            joints_quat_32, joints_quat_64, ...),
    motions=(motions_euler_32, motions_euler_64, ...,
             motions_original,
             motions_quat_32, motions_quat_64, ...),
    legends=('Euler win=32', 'Euler win=64', ...,
             'Original',
             'Quaternions win=32', 'Quaternions win=64', ...),
    legend_groups=('Euler', 'Euler', ...,
                   'Original',
                   'Quaternions', 'Quaternions', ...),
)
viewer.run()
```

