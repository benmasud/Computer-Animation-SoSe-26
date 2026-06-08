import os
import numpy as np
from transforms3d.euler import euler2quat, quat2euler
from motion_player.amc_parser import parse_asf, parse_amc
from motion_player.Viewer3D import Viewer


def extract_position_data(joints, motions):
    """Compute world-space 3D positions for all joints across all frames."""
    joint_names = list(joints.keys())
    n_frames    = len(motions)
    n_joints    = len(joint_names)
    positions   = np.zeros((n_frames, n_joints, 3), dtype=np.float64)

    for frame_idx, motion in enumerate(motions):
        joints['root'].set_motion(motion)
        for joint_idx, joint_name in enumerate(joint_names):
            joint = joints[joint_name]
            if joint.coordinate is not None:
                positions[frame_idx, joint_idx, :] = np.squeeze(joint.coordinate)
    return positions


def extract_rotation_euler_data(joints, motions):
    """Extract local Euler rotation angles (in degrees) for all joints across all frames."""
    joint_names = list(joints.keys())
    n_frames    = len(motions)
    n_joints    = len(joint_names)
    rotations   = np.zeros((n_frames, n_joints, 3), dtype=np.float64)

    for frame_idx, motion in enumerate(motions):
        for joint_idx, joint_name in enumerate(joint_names):
            joint = joints[joint_name]
            if joint_name == 'root':
                root_values = motion.get('root', [])
                if len(root_values) >= 6:
                    rotations[frame_idx, joint_idx, :] = root_values[3:6]
            else:
                values = motion.get(joint_name)
                if values is None or len(joint.dof) == 0:
                    continue
                joint_rot = np.zeros(3, dtype=np.float64)
                for channel_index, channel_name in enumerate(joint.dof):
                    if channel_name == 'rx':
                        joint_rot[0] = values[channel_index]
                    elif channel_name == 'ry':
                        joint_rot[1] = values[channel_index]
                    elif channel_name == 'rz':
                        joint_rot[2] = values[channel_index]
                rotations[frame_idx, joint_idx, :] = joint_rot
    return rotations


def euler_to_quaternion(euler_angles):
    """Convert Euler angles (in degrees) to a unit quaternion [w, x, y, z]."""
    rad = np.deg2rad(euler_angles)
    return euler2quat(rad[0], rad[1], rad[2], axes='sxyz')


def quaternion_to_euler(quaternion):
    """Convert a quaternion [w, x, y, z] to Euler angles in degrees."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    quaternion = quaternion / norm
    angles = quat2euler(quaternion, axes='sxyz')
    return np.rad2deg(angles)


def copy_skeleton_dict(skeleton):
    """Create an independent deep copy of a skeleton joint dictionary."""
    return skeleton['root'].copy().to_dict()


def run_assignment1(data_dir='data'):
    print('=' * 80)
    print('ASSIGNMENT 1 - Motion Capture Filtering')
    print('=' * 80)

    root_dir = os.path.abspath(os.path.dirname(__file__))
    asf_path = os.path.join(root_dir, data_dir, 'HDM_bk.asf')
    amc_path = os.path.join(root_dir, data_dir, 'HDM_bk_walk.amc')

    print(f'Loading skeleton: {asf_path}')
    print(f'Loading motion:   {amc_path}')
    joints  = parse_asf(asf_path)
    motions = parse_amc(amc_path)

    if len(motions) == 0:
        raise ValueError('No motion frames loaded from AMC.')

    print(f'Loaded {len(joints)} joints and {len(motions)} frames.')

    # TODO: implement assignment 1 here


if __name__ == '__main__':
    run_assignment1('data')