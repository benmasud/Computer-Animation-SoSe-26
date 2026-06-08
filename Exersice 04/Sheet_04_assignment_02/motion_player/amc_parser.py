
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D


class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.

    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.

    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.

    length: Length of the bone.

    axis: Axis of rotation for the bone.

    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.

    limits: Limits on each of the channels in the dof specification

    """
    self.name = name
    self.direction = np.reshape(direction, [3, 1])
    self.length = length
    axis = np.deg2rad(axis)
    self.C = euler2mat(*axis)
    self.Cinv = np.linalg.inv(self.C)
    self.dof = list(dof)
    self.limits = np.zeros([3, 2])
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None
    self.rootRotationalOffsetQuat = np.array([1, 0, 0, 0])  

  def copy(self):
    """
    Create an independent copy of this joint and all its children.
    This method properly handles the hierarchical structure of joints.
    
    Returns
    -------
    Joint: A new independent copy of this joint with all children recursively copied
    """
    new_joint = Joint(
      self.name,
      np.squeeze(self.direction),
      self.length,
      np.rad2deg(np.array([
        np.arctan2(self.C[2, 1], self.C[2, 2]),
        np.arcsin(-self.C[2, 0]),
        np.arctan2(self.C[1, 0], self.C[0, 0])
      ])),
      [],  
      []   
    )
    
    new_joint.dof = list(self.dof)
    new_joint.limits = np.copy(self.limits)

    if self.coordinate is not None:
      new_joint.coordinate = np.copy(self.coordinate)
    if self.matrix is not None:
      new_joint.matrix = np.copy(self.matrix)
    new_joint.rootRotationalOffsetQuat = np.copy(self.rootRotationalOffsetQuat)
    
    for child in self.children:
      new_child = child.copy()
      new_joint.children.append(new_child)
      new_child.parent = new_joint
    
    return new_joint

  def set_motion(self, motion):
    if self.name == 'root':
      self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])
      rotation = np.deg2rad(motion['root'][3:])
      self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
    else:
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = np.deg2rad(rotation)
      self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)
      self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
    for child in self.children:
      child.set_motion(motion)

  def draw(self):
    joints = self.to_dict()
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)

    xs, ys, zs = [], [], []
    for joint in joints.values():
      xs.append(joint.coordinate[0, 0])
      ys.append(joint.coordinate[1, 0])
      zs.append(joint.coordinate[2, 0])
    plt.plot(zs, xs, ys, 'b.')

    for joint in joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
        ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
        zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
        plt.plot(zs, xs, ys, 'r')
    plt.show()

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path, encoding="latin-1") as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':bonedata':
      content = content[idx+1:]
      break

  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
  idx = 0
  while True:

    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])

    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints


def parse_amc(file_path):
  with open(file_path, encoding="latin-1") as f:
    content = f.read().splitlines()

  marker_found = False
  start_idx = 0
  fully_spec = False
  
  for idx, line in enumerate(content):
    if line.strip().startswith('#'):
      start_idx = idx + 1
      continue
    
    line_upper = line.upper().strip()
    if line_upper == ':FULLY-SPECIFIED':
      start_idx = idx + 1
      fully_spec = True
      continue

    if line_upper == ':DEGREES':
      marker_found = True and fully_spec
      start_idx = idx + 1
      break
  
  if not marker_found:
    for idx, line in enumerate(content):
      if line.strip() and not line.strip().startswith('#'):
        start_idx = idx
        break
  
  content = content[start_idx:]

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  
  while line and line[0].startswith('#'):
    line, idx = read_line(content, idx)
  
  if not line or not line[0].isnumeric():
    raise ValueError(f"Expected frame number, got: {line}")
  
  EOF = False
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].startswith('#'):
        continue
      if line[0].isnumeric():
        break
      joint_degree[line[0]] = [float(deg) for deg in line[1:]]
    frames.append(joint_degree)
  return frames


def test_all():
  import os
  lv0 = './data'
  lv1s = os.listdir(lv0)
  for lv1 in lv1s:
    lv2s = os.listdir('/'.join([lv0, lv1]))
    asf_path = '%s/%s/%s.asf' % (lv0, lv1, lv1)
    print('parsing %s' % asf_path)
    joints = parse_asf(asf_path)
    motions = parse_amc('./nopose.amc')
    joints['root'].set_motion(motions[0])
    joints['root'].draw()



if __name__ == '__main__':
  test_all()
