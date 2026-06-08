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
    print('ASSIGNMENT 2: Spacetime Constraint Method')
  

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

    # Extract original hand positions
    print('\nExtracting original hand positions...')
    original_hand_positions = extract_joint_world_positions(joints, motions, 'rhand')
    print(f'Original hand Y range: [{original_hand_positions[:, 1].min():.2f}, {original_hand_positions[:, 1].max():.2f}]')
    
    # Define target trajectory: raise hand 5 units
    hand_offset = np.array([[0.0], [5.0], [0.0]], dtype=np.float64)
    target_trajectory = original_hand_positions.T + hand_offset
    print(f'Target hand Y range: [{target_trajectory[1].min():.2f}, {target_trajectory[1].max():.2f}]')

    # CONFIGURATION A: rhumerus only
    print('CONFIG A: Optimize rhumerus (upper arm only)')
    optimized_motions_A = optimizeWithConstraint(
        joints, motions, constraint=target_trajectory,
        joint_names=['rhumerus'], hand_name='rhand'
    )
    opt_hand_A = extract_joint_world_positions(joints, optimized_motions_A, 'rhand')
    print(f'Result Y range: [{opt_hand_A[:, 1].min():.2f}, {opt_hand_A[:, 1].max():.2f}]')

    # CONFIGURATION B: rhumerus + rradius
    print('CONFIG B: Optimize rhumerus + rradius (upper arm + forearm)')

    optimized_motions_B = optimizeWithConstraint(
        joints, motions, constraint=target_trajectory,
        joint_names=['rhumerus', 'rradius'], hand_name='rhand'
    )
    opt_hand_B = extract_joint_world_positions(joints, optimized_motions_B, 'rhand')
    print(f'Result Y range: [{opt_hand_B[:, 1].min():.2f}, {opt_hand_B[:, 1].max():.2f}]')

    # CONFIGURATION C: Full arm chain
    print('CONFIG C: Optimize rshoulder + rhumerus + rradius (full arm)')
    optimized_motions_C = optimizeWithConstraint(
        joints, motions, constraint=target_trajectory,
        joint_names=['rshoulder', 'rhumerus', 'rradius'], hand_name='rhand'
    )
    opt_hand_C = extract_joint_world_positions(joints, optimized_motions_C, 'rhand')
    print(f'Result Y range: [{opt_hand_C[:, 1].min():.2f}, {opt_hand_C[:, 1].max():.2f}]')

    # Visualize
    if show_viewer:
        print('Launching Viewer')
        viewer = Viewer(
            joints=(joints, joints, joints, joints),
            motions=(motions, optimized_motions_A, optimized_motions_B, optimized_motions_C),
            legends=('Original', 'Config A: rhumerus', 'Config B: arm', 'Config C: full arm'),
            legend_groups=('Original', 'Config A', 'Config B', 'Config C')
        )
        viewer.run()


if __name__ == '__main__':
    assignment2(max_frames=100, show_viewer=True)
