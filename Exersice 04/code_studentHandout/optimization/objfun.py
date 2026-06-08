import copy

import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm
from transforms3d.euler import euler2quat, quat2euler


def euler_to_quaternion(euler_angles):
    """Convert Euler angles [rx, ry, rz] in degrees to a normalized quaternion [w, x, y, z]."""
    rad = np.deg2rad(euler_angles)
    quat = euler2quat(rad[0], rad[1], rad[2], axes='sxyz')
    return normalize_quaternion(quat)


def quaternion_to_euler(quaternion):
    """Convert a quaternion [w, x, y, z] to Euler angles [rx, ry, rz] in degrees.

    The quaternion is normalized and its sign is canonicalized (w >= 0) before
    conversion to ensure a consistent Euler representation.
    """
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    q = q / norm
    # Canonicalize sign: ensure w >= 0 to avoid ambiguous Euler output
    if q[0] < 0:
        q = -q
    angles = quat2euler(q, axes='sxyz')
    return np.rad2deg(angles)


def normalize_quaternion(quaternion):
    """Normalize a quaternion to unit length and canonicalize its sign (w >= 0).

    Returns the identity quaternion [1, 0, 0, 0] if the input norm is near zero.
    """
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / norm
    # Enforce canonical form: q and -q represent the same rotation,
    # choosing w >= 0 makes the representation unique
    if q[0] < 0:
        q = -q
    return q


def motion_to_opt_motion(joints, motions):
    """Convert a list of AMC-style motion frames into the internal optimization structure.

    Extracts root translations and per-joint quaternion/Euler rotation arrays across
    all frames, then computes initial joint world positions via forward kinematics.
    The resulting dictionary is used by all optimization functions in this module.
    """
    joint_names = list(joints.keys())
    n_joints = len(joint_names)
    n_frames = len(motions)

    root_translation = np.zeros((3, n_frames), dtype=np.float64)
    rotationQuat  = [None] * n_joints
    rotationEuler = [None] * n_joints

    for joint_index, joint_name in enumerate(joint_names):
        quat_mat  = np.zeros((4, n_frames), dtype=np.float64)
        euler_mat = np.zeros((3, n_frames), dtype=np.float64)

        for frame_idx, motion in enumerate(motions):
            if joint_name == 'root':
                root_values = motion.get('root', [])
                if len(root_values) >= 3:
                    root_translation[:, frame_idx] = root_values[:3]
                euler = np.array(root_values[3:6], dtype=np.float64) if len(root_values) >= 6 else np.zeros(3)
            else:
                euler = np.zeros(3, dtype=np.float64)
                values = motion.get(joint_name, [])
                # Map each active DOF channel to the correct euler index
                for value, channel in zip(values, joints[joint_name].dof):
                    if channel == 'rx':
                        euler[0] = value
                    elif channel == 'ry':
                        euler[1] = value
                    elif channel == 'rz':
                        euler[2] = value

            quat_mat[:, frame_idx]  = euler_to_quaternion(euler)
            euler_mat[:, frame_idx] = euler

        rotationQuat[joint_index]  = quat_mat
        rotationEuler[joint_index] = euler_mat

    mot = {
        'njoints':          n_joints,
        'nframes':          n_frames,
        'frameTime':        1 / 120,
        'samplingRate':     120,
        'jointTrajectories': [None] * n_joints,
        'rootTranslation':  root_translation,
        'rotationEuler':    rotationEuler,
        'rotationQuat':     rotationQuat,
        'jointNames':       joint_names,
        'boneNames':        joint_names,
        'nameMap':          [],
        'animated':         [i for i, name in enumerate(joint_names) if len(joints[name].dof) > 0 or name == 'root'],
        'unanimated':       [i for i, name in enumerate(joint_names) if len(joints[name].dof) == 0 and name != 'root'],
        'boundingBox':      [],
        'filename':         '',
        'documentation':    '',
        'angleUnit':        'deg'
    }

    # Compute initial world-space joint positions
    mot['jointTrajectories'] = forwardKinematicsQuat(joints, mot)
    return mot


def apply_optimized_frame_to_motion(frame, rFrame, oFrame, joints, jointNames, jointList):
    """Write optimized quaternion rotations back into an AMC-style frame dictionary.

    Converts the optimized quaternion for each joint in jointList back to Euler
    angles and updates the corresponding entry in the AMC frame. A sign correction
    is applied before conversion because q and -q represent the same rotation but
    produce completely different Euler angles.
    """
    for jointIndex in jointList:
        joint_name = jointNames[jointIndex]
        joint = joints[joint_name]
        if joint_name == 'root':
            continue

        quat          = rFrame['rotationQuat'][jointIndex]
        original_quat = oFrame['rotationQuat'][jointIndex]

        # q and -q are the same rotation but produce different Euler angles —
        # align sign with the original quaternion before converting
        if np.dot(original_quat, quat) < 0:
            quat = -quat

        euler = quaternion_to_euler(quat)

        # Only write back channels that are active for this joint
        values = []
        for channel in joint.dof:
            if channel == 'rx':
                values.append(float(euler[0]))
            elif channel == 'ry':
                values.append(float(euler[1]))
            elif channel == 'rz':
                values.append(float(euler[2]))
        if values:
            frame[joint_name] = values

    return frame


def objfun(x, oFrame, skel, constraints, optProps, jointList):
    """Compute the residual vector for the spacetime constraint optimization.

    Called repeatedly by least_squares() with candidate solutions. Returns a
    concatenated vector of three weighted error terms that least_squares minimizes
    by adjusting the joint quaternions in x.

    Parameters
    ----------
    x : np.ndarray
        Current optimization vector — flattened quaternions for all joints in jointList.
    oFrame : dict
        Original (unmodified) frame data, used as reference for orientation and
        smoothness terms.
    skel : dict
        Skeleton joint dictionary as returned by parse_asf().
    constraints : list
        Target world position for the hand joint, shape (3,).
    optProps : dict
        Optimization state, including 'handName' and 'x_1' (previous frame solution).
    jointList : list of int
        Indices of joints included in the optimization.

    Returns
    -------
    np.ndarray
        Residual vector to be minimized by least_squares(). Concatenation of:
        - Term 1: hand position error (weighted by weightHandPosition)
        - Term 2: orientation error  (weighted by weightOrientation)
        - (optionally) Term 3: smoothness error   (weighted by weightSmoothness)
    """
    # TODO: implement the objective function
    # Hints:
    # - Use X2Frame(x, copy.deepcopy(oFrame), jointList) to reconstruct the frame
    # - Use forwardKinematicsQuat(skel, rFrame) to compute joint world positions
    # - Term 1: position error — distance of hand joint to target constraint
    # - Term 2: orientation error — keep optimized joints close to original rotation
    # - Combine both terms with appropriate weights and return as a single vector
    raise NotImplementedError("Implement the objective function here.")


def forwardKinematicsQuat(skel, mot):
    """Compute world-space joint positions for all joints and frames using quaternion rotations.

    Starts from the root joint and traverses the skeleton hierarchy recursively.
    Returns a list of position arrays, one per joint, each of shape (3, nframes).
    """
    root_joint  = skel['root']
    root_offset = np.array(root_joint.direction).flatten() * root_joint.length

    # Initial root position: translation + bone offset
    initial_position = mot['rootTranslation'] + np.tile(
        root_offset.reshape(-1, 1), (1, mot['nframes'])
    )

    # Apply the root joint's rotational offset (if any)
    root_rot_offset  = root_joint.rootRotationalOffsetQuat
    root_rotation    = mot['rotationQuat'][0]
    initial_rotation = quatmult(
        np.tile(root_rot_offset.reshape(-1, 1), (1, mot['nframes'])),
        root_rotation
    )

    trajectories = [None] * mot['njoints']
    return recursive_forwardKinematicsQuat(
        skel, mot, 'root', initial_position, initial_rotation, trajectories
    )


def recursive_forwardKinematicsQuat(skel, mot, node_name, current_position, current_rotation, trajectories):
    """Recursively compute world-space positions for a joint and all its descendants.

    For each child, the accumulated rotation is the parent rotation composed with
    the child's local rotation. The child's world position is the parent position
    plus the bone offset vector rotated into world space.
    """
    node_id = list(skel.keys()).index(node_name)
    trajectories[node_id] = current_position

    joint = skel[node_name]
    for child in joint.children:
        child_name = child.name
        child_idx  = list(skel.keys()).index(child_name)

        # Compose parent rotation with child's local rotation
        if mot['rotationQuat'][child_idx] is not None:
            child_rotation = quatmult(current_rotation, mot['rotationQuat'][child_idx])
        else:
            child_rotation = current_rotation

        # Rotate the bone direction vector into world space and add to parent position
        child_offset   = np.array(child.direction).flatten() * child.length
        child_position = current_position + quatrot(
            np.tile(child_offset.reshape(-1, 1), (1, mot['nframes'])),
            child_rotation
        )
        trajectories = recursive_forwardKinematicsQuat(
            skel, mot, child_name, child_position, child_rotation, trajectories
        )

    return trajectories


def quatmult(q1, q2):
    """Multiply two quaternions q1 and q2 (Hamilton product).

    Supports broadcasting: q1 and q2 can be shape (4,) or (4, n).
    Returns the composed rotation q1 * q2.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float64)


def quatrot(v, q):
    """Rotate a 3D vector (or batch of vectors) v by quaternion q.

    Implements the sandwich product q * [0, v] * q_conj and returns the
    rotated vector(s). v has shape (3, n), q has shape (4, n).
    """
    q_norm = q / np.linalg.norm(q, axis=0)
    q_conj = np.array([q_norm[0], -q_norm[1], -q_norm[2], -q_norm[3]])

    # Embed v as a pure quaternion [0, vx, vy, vz]
    v_quat = np.vstack([np.zeros(v.shape[1], dtype=np.float64), v])
    temp   = quatmult(q_norm, v_quat)
    result = quatmult(temp, q_conj)
    return result[1:4]


def X2Frame(X, f, jointList):
    """Unpack the optimization vector X into the rotationQuat entries of frame f.

    Each group of 4 values in X corresponds to the quaternion of one joint in
    jointList. The quaternion is normalized and sign-corrected relative to the
    original to stay on the same hemisphere.
    """
    idx = 0
    for jointIndex in jointList:
        if f['rotationQuat'][jointIndex] is not None:
            quat = np.asarray(X[idx:idx + 4], dtype=np.float64)
            quat = normalize_quaternion(quat)
            originalQuat = f['rotationQuat'][jointIndex]
            # Keep the optimized quaternion on the same hemisphere as the original
            # to avoid discontinuities in the orientation penalty
            if np.dot(originalQuat, quat) < 0:
                quat = -quat
            f['rotationQuat'][jointIndex] = quat
            idx += 4
    return f


def Frame2X(frame, jointList):
    """Pack the rotationQuat entries of a frame into a flat optimization vector.

    Inverse of X2Frame. Concatenates the quaternion [w, x, y, z] of each joint
    in jointList into a single 1D array suitable for least_squares().
    """
    res = []
    for jointIndex in jointList:
        if frame['rotationQuat'][jointIndex] is not None:
            res.extend(frame['rotationQuat'][jointIndex])
    return np.array(res, dtype=np.float64)


def extractFrame(mot, f):
    """Extract a single frame f (1-indexed) from the optimization motion structure.

    Returns a single-frame motion dict with the same structure as mot but
    containing only the data for the requested frame index.
    """
    frame = {
        'njoints':          mot['njoints'],
        'nframes':          1,
        'frameTime':        mot['frameTime'],
        'samplingRate':     mot['samplingRate'],
        'jointTrajectories': [traj[:, f - 1] if traj is not None else None for traj in mot['jointTrajectories']],
        'rotationQuat':     [quat[:, f - 1] if quat is not None else None for quat in mot['rotationQuat']],
        'rootTranslation':  mot['rootTranslation'][:, f - 1],
        'jointNames':       mot['jointNames']
    }
    return frame


def addFrame2Motion(mot, Frame):
    """Append a single-frame motion dict to a multi-frame motion structure.

    Horizontally stacks joint trajectories, quaternion arrays, and root
    translation. Used to accumulate optimized frames into a full sequence.
    """
    mot['nframes'] += Frame['nframes']
    for j in range(mot['njoints']):
        if mot['jointTrajectories'][j] is not None and Frame['jointTrajectories'][j] is not None:
            mot['jointTrajectories'][j] = np.hstack([mot['jointTrajectories'][j], Frame['jointTrajectories'][j]])
        elif mot['jointTrajectories'][j] is None:
            mot['jointTrajectories'][j] = Frame['jointTrajectories'][j]
        if mot['rotationQuat'][j] is not None and Frame['rotationQuat'][j] is not None:
            mot['rotationQuat'][j] = np.hstack([mot['rotationQuat'][j], Frame['rotationQuat'][j]])
        elif mot['rotationQuat'][j] is None:
            mot['rotationQuat'][j] = Frame['rotationQuat'][j]
    mot['rootTranslation'] = np.hstack([
        mot['rootTranslation'],
        Frame['rootTranslation'].reshape(3, Frame['nframes'])
    ])
    return mot


def optimizeWithConstraint(skel, motions, constraint=None, joint_names=None, hand_name='rhand'):
    """Run the spacetime constraint optimization frame by frame and return optimized AMC motions."""
    mot = motion_to_opt_motion(skel, motions)
    if joint_names is None:
        joint_names = ['rhumerus']

    # Convert joint names to indices used throughout the optimization
    jointList = [mot['jointNames'].index(name) for name in joint_names if name in mot['jointNames']]
    if not jointList:
        raise ValueError(f'No valid joints found for optimization: {joint_names}')

    # Default constraint: raise the hand 5 units above its original trajectory
    if constraint is None:
        hand_index = mot['jointNames'].index(hand_name)
        offset     = np.array([[0.0], [5.0], [0.0]], dtype=np.float64)
        constraint = mot['jointTrajectories'][hand_index] + np.tile(offset, (1, mot['nframes']))

    constraint = np.asarray(constraint, dtype=np.float64)
    if constraint.shape != (3, mot['nframes']):
        raise ValueError('Constraint must have shape (3, nframes)')

    optProps = {
        'Display':    'iter',
        'startValue': [],
        'Iterations': 50,
        'FuncEvals':  10000,   # max function evaluations per frame
        'TolFun':     0.01,    # convergence tolerance on residual
        'TolFunX':    0.01,    # convergence tolerance on step size
        'x':          [],
        'x_1':        [],      # solution from previous frame (warm start)
        'x_2':        [],      # solution from two frames ago
        'recmot':     [],
        'handName':   hand_name
    }

    optimized_motions = []
    with tqdm(total=mot['nframes'], desc="Optimizing frames", unit="frame") as pbar:
        for frame_idx in range(1, mot['nframes'] + 1):
            oFrame = extractFrame(mot, frame_idx)
            target = constraint[:, frame_idx - 1]
            rFrame, optProps, cost = constructFrameWithConstraint(
                skel, oFrame, [target], optProps, jointList
            )
            optimized_frame = apply_optimized_frame_to_motion(
                copy.deepcopy(motions[frame_idx - 1]),
                rFrame,
                oFrame,
                skel,
                mot['jointNames'],
                jointList
            )
            optimized_motions.append(optimized_frame)
            pbar.set_postfix({"frame": frame_idx, "cost": f"{cost:.2f}"})
            pbar.update(1)

    return optimized_motions


def constructFrameWithConstraint(skel, oFrame, constraint, optProps, jointList):
    """Optimize a single frame to satisfy the hand position constraint using least_squares()."""
    # Use previous frame's solution as starting point (warm start),
    # or fall back to the original frame quaternions on the first frame
    if optProps['x_1'] is not None and len(optProps['x_1']) > 0:
        x0 = optProps['x_1']
    else:
        x0 = Frame2X(oFrame, jointList)

    def fun(x):
        return objfun(x, oFrame, skel, constraint, optProps, jointList)

    result = least_squares(
        fun,
        x0,
        bounds=(-1.0, 1.0),   # quaternion components are bounded to [-1, 1]
        max_nfev=optProps['FuncEvals'],
        ftol=optProps['TolFun'],
        xtol=optProps['TolFunX'],
        verbose=0,
    )

    X = result.x
    optProps['startValue'] = X
    optProps['x_2'] = optProps['x_1']  # shift history back by one frame
    optProps['x_1'] = X                # store current solution for next frame

    # Reconstruct the optimized frame and compute final joint positions
    rFrame = X2Frame(X, copy.deepcopy(oFrame), jointList)
    rFrame['jointTrajectories'] = forwardKinematicsQuat(skel, rFrame)
    return rFrame, optProps, result.cost


def emptyMotion(refmot=None):
    """Create an empty motion structure, optionally shaped after a reference motion.

    If refmot is None, returns a minimal empty structure with zero joints and frames.
    If refmot is provided, returns a zero-initialized structure with the same
    dimensions and metadata as refmot, useful as a blank canvas for building
    new motion sequences.
    """
    if refmot is None:
        mot = {
            'njoints':          0,
            'nframes':          0,
            'frameTime':        1 / 120,
            'samplingRate':     120,
            'jointTrajectories': [],
            'rootTranslation':  np.array([]),
            'rotationEuler':    [],
            'rotationQuat':     [],
            'jointNames':       [],
            'boneNames':        [],
            'nameMap':          [],
            'animated':         [],
            'unanimated':       [],
            'boundingBox':      [],
            'filename':         '',
            'documentation':    '',
            'angleUnit':        'deg'
        }
    else:
        mot = {
            'njoints':          refmot['njoints'],
            'nframes':          refmot['nframes'],
            'frameTime':        refmot['frameTime'],
            'samplingRate':     refmot['samplingRate'],
            'jointTrajectories': [None] * refmot['njoints'],
            'rootTranslation':  np.zeros((3, refmot['nframes'])),
            'rotationEuler':    [None] * refmot['njoints'],
            'rotationQuat':     [None] * refmot['njoints'],
            'jointNames':       refmot['jointNames'],
            'boneNames':        refmot['boneNames'],
            'nameMap':          refmot['nameMap'],
            'animated':         refmot['animated'],
            'unanimated':       refmot['unanimated'],
            'boundingBox':      [],
            'filename':         '',
            'documentation':    '',
            'angleUnit':        'deg'
        }
    return mot