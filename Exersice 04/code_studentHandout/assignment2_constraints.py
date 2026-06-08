import os
import numpy as np
from motion_player.amc_parser import parse_asf, parse_amc
from motion_player.Viewer3D import Viewer
from optimization import optimizeWithConstraint


def copy_skeleton_dict(joints):
    """Create an independent deep copy of a skeleton joint dictionary."""
    return joints['root'].copy().to_dict()


def extract_joint_world_positions(joints, motions, joint_name):
    """Extract world-space 3D positions for a single named joint across all frames."""
    positions = np.zeros((len(motions), 3), dtype=np.float64)
    skeleton  = copy_skeleton_dict(joints)
    for frame_idx, motion in enumerate(motions):
        skeleton['root'].set_motion(motion)
        positions[frame_idx, :] = np.squeeze(skeleton[joint_name].coordinate)
    return positions


def assignment2(max_frames=None, show_viewer=True):
    print('=' * 80)
    print('ASSIGNMENT 2: Spacetime Constraint Method')
    print('=' * 80)

    root_dir = os.path.abspath(os.path.dirname(__file__))
    asf_path = os.path.join(root_dir, 'data', 'HDM_bk.asf')
    amc_path = os.path.join(root_dir, 'data', 'HDM_bk_walk.amc')

    print(f'Loading skeleton: {asf_path}')
    print(f'Loading motion:   {amc_path}')
    joints  = parse_asf(asf_path)
    motions = parse_amc(amc_path)

    if len(motions) == 0:
        raise ValueError('No motion frames loaded from AMC.')

    print(f'Loaded {len(joints)} joints and {len(motions)} frames.')

    if max_frames is not None:
        motions = motions[:max_frames]
        print(f'Using first {max_frames} frames for testing.')

    # TODO: implement assignment 2 here
    # - Define a target hand trajectory (constraint) based on the original hand positions
    # - Call optimizeWithConstraint() to optimize the motion
    # - Visualize original and optimized motion in the viewer
    # - Experiment with different joint_names and weights in objfun (see optimization/objfun.py)


if __name__ == '__main__':
    assignment2()