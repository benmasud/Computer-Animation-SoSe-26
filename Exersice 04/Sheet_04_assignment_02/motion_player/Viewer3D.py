import pygame
import numpy as np
import transforms3d.euler as euler

from OpenGL.GL import *
from OpenGL.GLU import *


class Viewer:
  def __init__(self, joints=None, motions=None, legends=None, colors=None, legend_groups=None):
    """
    Display one or multiple motion capture sequences side-by-side in 3D.

    All skeletons are rendered overlaid in the same scene, each in a distinct
    color. Visibility of individual skeletons can be toggled via the legend
    panel. When legend_groups is provided, skeletons are split into up to three
    separate legend panels (left, center, right) for easier comparison.

    Parameters
    ----------
    joints : dict or sequence of dict
        One skeleton per entry, as returned by parse_asf(). A single dict is
        automatically wrapped in a tuple.
    motions : list or sequence of list
        One motion sequence per entry, as returned by parse_amc(). A single
        list is automatically wrapped in a tuple.
    legends : sequence of str or None
        Display label for each skeleton. Length must match number of skeletons.
        Defaults to 'skel1', 'skel2', ... if not provided.
    colors : sequence of (r, g, b) tuples or None
        RGB colors for each skeleton, with components in [0, 1]. Length must
        match number of skeletons. If not provided, a default color palette is
        used automatically.
    legend_groups : sequence of str or None
        Group label for each skeleton, e.g. ['Euler', 'Euler', 'Quat', 'Quat'].
        The first unique group is placed in the left panel, the second in the
        center, the third in the right panel. If None, all skeletons appear in
        a single right-side panel (original behavior).
    """

    # --- Normalize joints and motions to tuples ---
    if joints is None:
      self.joints = tuple()
    elif isinstance(joints, dict):
      self.joints = (joints,)
    else:
      self.joints = tuple(joints)

    if motions is None:
      self.motions = tuple()
    elif isinstance(motions, list):
      self.motions = (motions,)
    else:
      self.motions = tuple(motions)

    if len(self.joints) != len(self.motions):
      raise ValueError(
        f"Number of joint sets ({len(self.joints)}) must equal number of motion sequences ({len(self.motions)})."
      )

    # Validate each skeleton's motion data before opening the window
    for idx, (joints, motions) in enumerate(zip(self.joints, self.motions)):
      self._validate_motion_data(idx, joints, motions)

    self.num_skeletons = len(self.joints)

    if self.num_skeletons == 0:
      self.joints = tuple()
      self.motions = tuple()

    # --- Legends ---
    if legends is None:
      legends = [f"skel{idx+1}" for idx in range(self.num_skeletons)]
    legends = tuple(legends)
    if len(legends) != self.num_skeletons:
      raise ValueError(
        f"Length of legends ({len(legends)}) must match number of skeletons ({self.num_skeletons})."
      )
    self.legends = legends

    # --- Legend groups (optional) ---
    # Controls which panel each skeleton's legend entry appears in.
    if legend_groups is not None:
      if len(legend_groups) != self.num_skeletons:
        raise ValueError(
          f"Length of legend_groups ({len(legend_groups)}) must match number of skeletons ({self.num_skeletons})."
        )
    self.legend_groups = legend_groups

    # --- Colors ---
    if colors is not None:
      colors = tuple(colors)
      if len(colors) != self.num_skeletons:
        raise ValueError(
          f"Length of colors ({len(colors)}) must match number of skeletons ({self.num_skeletons})."
        )
      for c in colors:
        if not (isinstance(c, (tuple, list)) and len(c) == 3):
          raise TypeError("Each color must be a tuple/list of 3 floats in [0, 1].")
        if not all(0.0 <= v <= 1.0 for v in c):
          raise ValueError("RGB color components must be in [0, 1].")
      self.skeleton_colors = list(colors)
    else:
      self.skeleton_colors = self.generate_default_colors(self.num_skeletons)

    # All skeletons start visible
    self.skeleton_visibility = [True] * self.num_skeletons

    # --- Playback state ---
    self.frame = 0
    self.playing = False
    self.fps = 120

    # --- Camera / interaction state ---
    self.rotate_dragging = False
    self.translate_dragging = False
    self.old_x = 0
    self.old_y = 0
    self.global_rx = 0
    self.global_ry = 0
    self.rotation_R = np.eye(3)
    self.speed_rx = np.pi / 90
    self.speed_ry = np.pi / 90
    self.speed_trans = 0.25
    self.speed_zoom = 0.5
    self.done = False
    self.default_translate = np.array([0, 0, -150], dtype=np.float32)
    self.translate = np.copy(self.default_translate)

    # --- Pygame / OpenGL window setup ---
    pygame.init()
    info = pygame.display.Info()
    self.screen_size = (info.current_w, info.current_h)
    self.screen = pygame.display.set_mode(
      self.screen_size, pygame.DOUBLEBUF | pygame.OPENGL
    )
    max_frames = max([len(m) for m in self.motions if m]) if self.motions else 0
    pygame.display.set_caption(
      'AMC Parser - frame %d / %d (%d skeletons)' % (self.frame, max_frames, self.num_skeletons)
    )
    self.clock = pygame.time.Clock()

    # --- UI state ---
    self.input_active = False
    self.frame_input = ""
    self.slider_dragging = False

    # --- UI layout: bottom bar with playback controls and slider ---
    self.button_height = 30
    self.button_width = 60
    self.ui_margin = 10
    self.ui_y = self.screen_size[1] - self.button_height - self.ui_margin

    available_width = self.screen_size[0] - 2 * self.ui_margin

    # Slider takes 70% of the width; remaining space is split among controls
    slider_percent = 0.70
    self.slider_width = int(available_width * slider_percent)

    remaining_width = available_width - self.slider_width
    element_width = remaining_width / 6

    x_pos = self.ui_margin

    self.button_prev  = pygame.Rect(x_pos, self.ui_y, self.button_width, self.button_height)
    x_pos += element_width

    self.button_play  = pygame.Rect(x_pos, self.ui_y, self.button_width, self.button_height)
    x_pos += element_width

    self.button_next  = pygame.Rect(x_pos, self.ui_y, self.button_width, self.button_height)
    x_pos += element_width

    self.frame_label_x = x_pos
    x_pos += element_width

    self.input_width  = int(element_width * 0.9)
    self.input_field  = pygame.Rect(x_pos, self.ui_y, self.input_width, self.button_height)
    x_pos += element_width

    self.slider_x_start      = x_pos
    self.slider_height       = 8
    self.slider_y            = self.ui_y + (self.button_height - self.slider_height) // 2
    self.slider_rect         = pygame.Rect(self.slider_x_start, self.slider_y, self.slider_width, self.slider_height)
    self.slider_handle_radius = 6
    x_pos += self.slider_width + 10

    self.button_quit = pygame.Rect(x_pos, self.ui_y, self.button_width, self.button_height)

    pygame.font.init()
    self.font = pygame.font.Font(None, 24)

    # Legend buttons are rebuilt every frame in draw_legend()
    self.legend_buttons      = []
    self.legend_button_width  = 50
    self.legend_button_height = 20

    # Compute floor Y from the first frame of each skeleton
    self.floor_y = self.calculate_skeleton_floor_position()
    print(f"Floor Y position set to: {self.floor_y}")

    self.initial_root_positions = self.calculate_root_offset()

    # --- OpenGL material and rendering settings ---
    glClearColor(1, 1, 1, 1)
    glShadeModel(GL_SMOOTH)
    glMaterialfv(GL_FRONT, GL_SPECULAR, np.array([1, 1, 1, 1],     dtype=np.float32))
    glMaterialfv(GL_FRONT, GL_SHININESS, np.array([100.0],          dtype=np.float32))
    glMaterialfv(GL_FRONT, GL_AMBIENT,   np.array([0.7, 0.7, 0.7, 0.7], dtype=np.float32))
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32))

    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    gluPerspective(45, (self.screen_size[0] / self.screen_size[1]), 0.1, 500.0)

    glPointSize(10)
    glLineWidth(2.5)


  @staticmethod
  def generate_default_colors(n):
    """
    Generate n visually distinct RGB colors for OpenGL rendering.

    Colors are drawn from a fixed palette of 8 entries and cycled if n > 8.

    Parameters
    ----------
    n : int
        Number of colors to generate.

    Returns
    -------
    list of (r, g, b) tuples
        Colors with components in [0, 1].
    """
    base_colors = [
      (1.0, 0.0, 0.0),  # red
      (0.0, 1.0, 0.0),  # green
      (0.0, 0.0, 1.0),  # blue
      (1.0, 1.0, 0.0),  # yellow
      (1.0, 0.0, 1.0),  # magenta
      (0.0, 1.0, 1.0),  # cyan
      (0.5, 0.5, 0.5),  # gray
      (1.0, 0.5, 0.0),  # orange
    ]
    if n <= len(base_colors):
      return base_colors[:n]
    else:
      return [base_colors[i % len(base_colors)] for i in range(n)]


  def _validate_motion_data(self, skeleton_idx, joints, motions):
    """
    Validate motion data for a single skeleton before rendering starts.

    Checks that joints and motions are non-None, that every frame is a dict
    containing a 'root' key with numeric position data, and that all animated
    joints (those with at least one DOF) are present in every frame.

    Parameters
    ----------
    skeleton_idx : int
        Index used in error messages to identify the offending skeleton.
    joints : dict
        Joint hierarchy as returned by parse_asf().
    motions : list of dict
        Motion frames as returned by parse_amc().

    Raises
    ------
    ValueError
        If any structural or content check fails.
    TypeError
        If motions is not a list or tuple, or a frame is not a dict.
    """
    if joints is None:
      raise ValueError(f"Skeleton {skeleton_idx}: joints cannot be None")

    if motions is None:
      raise ValueError(f"Skeleton {skeleton_idx}: motions cannot be None")

    if not isinstance(motions, (list, tuple)):
      raise TypeError(f"Skeleton {skeleton_idx}: motions must be a list or tuple, got {type(motions)}")

    if len(motions) == 0:
      raise ValueError(f"Skeleton {skeleton_idx}: motions cannot be empty")

    for frame_idx, frame in enumerate(motions):
      if not isinstance(frame, dict):
        raise TypeError(f"Skeleton {skeleton_idx}, frame {frame_idx}: each frame must be a dict, got {type(frame)}")

      if 'root' not in frame:
        raise ValueError(f"Skeleton {skeleton_idx}, frame {frame_idx}: 'root' joint data missing")

      root_data = frame['root']
      if not isinstance(root_data, (list, tuple)) or len(root_data) < 3:
        raise ValueError(
          f"Skeleton {skeleton_idx}, frame {frame_idx}: root data must be list/tuple with at least 3 elements"
        )

      try:
        float(root_data[0]), float(root_data[1]), float(root_data[2])
      except (ValueError, TypeError):
        raise ValueError(f"Skeleton {skeleton_idx}, frame {frame_idx}: root position values must be numeric")

    # Check that all joints with active DOFs appear in every frame
    expected_animated_joints = {
      name for name, joint in joints.items() if len(joint.dof) > 0
    }

    for frame_idx, frame in enumerate(motions):
      frame_joints = set(frame.keys())
      if not expected_animated_joints.issubset(frame_joints):
        missing = expected_animated_joints - frame_joints
        extra   = frame_joints - expected_animated_joints
        error_msg = f"Skeleton {skeleton_idx}, frame {frame_idx}: joint mismatch."
        if missing:
          error_msg += f" Missing animated joints: {missing}"
        if extra:
          error_msg += f" Unexpected joints: {extra}"
        raise ValueError(error_msg)

    print(f"Skeleton {skeleton_idx}: validated {len(motions)} frames with {len(joints)} joints")


  def calculate_skeleton_floor_position(self):
    """
    Determine the floor Y coordinate from the lowest joint position in the first frame.

    The floor plane is rendered at this Y value so that the skeleton appears
    to stand on it rather than float above or sink below.

    Returns
    -------
    float
        The minimum Y coordinate found across all skeletons, or 0 if unavailable.
    """
    if not self.joints or not self.motions:
      return 0

    min_y_values = []
    for joints, motions in zip(self.joints, self.motions):
      if joints is None or motions is None or len(motions) == 0:
        continue

      joints['root'].set_motion(motions[0])

      min_y = float('inf')
      for joint in joints.values():
        if joint.coordinate is not None:
          y_coord = joint.coordinate[1, 0]
          if y_coord < min_y:
            min_y = y_coord

      if min_y != float('inf'):
        min_y_values.append(min_y)

    return min(min_y_values) if min_y_values else 0


  def switch_to_2d(self):
    """
    Switch the OpenGL projection matrix to orthographic 2D mode for UI rendering.

    Saves the current 3D projection and modelview matrices on the stack so they
    can be restored by switch_to_3d(). Also disables lighting and depth testing,
    which are not needed for flat UI elements.
    """
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, self.screen_size[0], self.screen_size[1], 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)


  def switch_to_3d(self):
    """
    Restore the OpenGL projection matrix to perspective 3D mode after UI rendering.

    Pops the matrices saved by switch_to_2d() and re-enables lighting and depth
    testing for skeleton rendering.
    """
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


  def draw_floor(self):
    """
    Draw a semi-transparent floor grid for spatial reference.

    The floor quad is placed at the Y level computed by calculate_skeleton_floor_position()
    and rotated together with the scene so it stays aligned with the skeleton.
    A grid of lines is drawn on top of the quad to aid depth perception.
    """
    floor_size = 100
    floor_y    = self.floor_y
    alpha      = 0.3

    glDisable(GL_LIGHTING)

    # Solid floor quad
    glColor4f(0.8, 0.8, 0.8, alpha)
    vertices = [
      np.array([-floor_size, floor_y, -floor_size]),
      np.array([ floor_size, floor_y, -floor_size]),
      np.array([ floor_size, floor_y,  floor_size]),
      np.array([-floor_size, floor_y,  floor_size]),
    ]
    glBegin(GL_QUADS)
    for v in vertices:
      transformed = v.dot(self.rotation_R) + self.translate
      glVertex3f(*transformed.astype(np.float32))
    glEnd()

    # Grid lines
    glColor4f(0.6, 0.6, 0.6, 0.2)
    glBegin(GL_LINES)
    grid_step = 10
    for i in range(-floor_size, floor_size + 1, grid_step):
      p1 = np.array([i, floor_y, -floor_size]).dot(self.rotation_R) + self.translate
      p2 = np.array([i, floor_y,  floor_size]).dot(self.rotation_R) + self.translate
      glVertex3f(*p1.astype(np.float32))
      glVertex3f(*p2.astype(np.float32))

      p3 = np.array([-floor_size, floor_y, i]).dot(self.rotation_R) + self.translate
      p4 = np.array([ floor_size, floor_y, i]).dot(self.rotation_R) + self.translate
      glVertex3f(*p3.astype(np.float32))
      glVertex3f(*p4.astype(np.float32))
    glEnd()

    glEnable(GL_LIGHTING)


  def clamp_frame(self, frame):
    """
    Clamp a frame number to the valid range [0, max_frames - 1].

    Parameters
    ----------
    frame : int
        Frame number to clamp.

    Returns
    -------
    int
        Clamped frame number.
    """
    max_frames = max(
      (len(m) for m in self.motions if m is not None),
      default=0
    )
    if max_frames == 0:
      return 0
    return max(0, min(frame, max_frames - 1))


  def check_button_click(self, pos, rect):
    """
    Return True if the given screen position falls inside the given rectangle.

    Parameters
    ----------
    pos : tuple of (int, int)
        Mouse position in screen coordinates.
    rect : pygame.Rect
        Button area to test against.

    Returns
    -------
    bool
    """
    return rect.collidepoint(pos)


  def get_slider_frame_from_pos(self, x):
    """
    Convert a horizontal mouse position to a frame number via the slider geometry.

    Parameters
    ----------
    x : int
        Horizontal mouse position in screen coordinates.

    Returns
    -------
    int
        Frame number corresponding to the slider position, clamped to valid range.
    """
    max_frames = max(
      (len(m) for m in self.motions if m is not None),
      default=0
    )
    if max_frames == 0:
      return 0

    relative_x = max(0, min(x - self.slider_x_start, self.slider_width))
    frame = int((relative_x / self.slider_width) * (max_frames - 1))
    return self.clamp_frame(frame)


  def draw_ui(self):
    """
    Draw all 2D UI elements as an overlay on top of the 3D scene.

    Switches to orthographic projection, draws the bottom control bar
    (previous/play/next buttons, frame input, slider, quit button) and
    the skeleton legend panels, then switches back to perspective projection.
    """
    self.switch_to_2d()

    # Dark semi-transparent background strip at the bottom
    glColor4f(0.2, 0.2, 0.2, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(0,                    self.ui_y - 5)
    glVertex2f(self.screen_size[0],  self.ui_y - 5)
    glVertex2f(self.screen_size[0],  self.screen_size[1])
    glVertex2f(0,                    self.screen_size[1])
    glEnd()

    self.draw_button(self.button_prev, "<<",   (0.3, 0.7, 0.3))

    button_text = "Stop" if self.playing else "Play"
    self.draw_button(self.button_play, button_text, (0.3, 0.5, 0.8))

    self.draw_button(self.button_next, ">>",   (0.3, 0.7, 0.3))
    self.draw_text("Frame:", self.frame_label_x + 5, self.input_field.y + 5, (1, 1, 1))
    self.draw_input_field()
    self.draw_slider()

    # Frame counter (top-right)
    max_frames = max((len(m) for m in self.motions if m is not None), default=0)
    frame_text   = "%d / %d" % (self.frame, max_frames)
    text_surface = self.font.render(frame_text, True, (0, 0, 0))
    self.draw_text(frame_text, self.screen_size[0] - text_surface.get_width() - 10, 10, (0, 0, 0))

    # Skeleton count (top-left)
    self.draw_text("Skeletons: %d" % self.num_skeletons, 10, 10, (0, 0, 0))

    self.draw_button(self.button_quit, "Quit", (0.8, 0.3, 0.3))
    self.draw_legend()

    self.switch_to_3d()


  def draw_legend(self):
    """
    Draw skeleton legend panels with colored labels and ON/OFF toggle buttons.

    If legend_groups is None, all skeletons are listed in a single right-side
    panel. If legend_groups is set, skeletons are split into up to three panels
    positioned at the left, center, and right of the screen. Each panel shows
    a group title, a colored stripe per skeleton, its label, and a toggle button.

    Toggle buttons update skeleton_visibility, which controls rendering in draw().
    """
    if self.legends is None:
      return

    margin        = 10
    line_length   = 40
    line_height   = 4
    spacing_y     = 25
    button_margin = 5

    # Rebuild button list every frame (screen layout may change)
    self.legend_buttons       = []
    self.legend_button_width  = 50
    self.legend_button_height = 20

    # Determine panel layout from legend_groups
    if self.legend_groups is not None:
      unique_groups = list(dict.fromkeys(self.legend_groups))  # preserve insertion order
      sides  = ['left', 'center', 'right']
      panels = [
        (sides[i] if i < len(sides) else 'right',
         group,
         [j for j, g in enumerate(self.legend_groups) if g == group])
        for i, group in enumerate(unique_groups)
      ]
    else:
      panels = [('right', None, list(range(self.num_skeletons)))]

    for (side, group_name, indices) in panels:
      # Compute the widest label in this panel to align the toggle buttons
      max_text_width = max(
        (self.font.render(self.legends[idx] or "", True, (0, 0, 0)).get_width()
         for idx in indices),
        default=0
      )
      total_width  = line_length + 8 + max_text_width + self.legend_button_width + button_margin * 2
      title_offset = 20 if group_name else 0  # extra vertical space for the group title

      # Horizontal anchor for this panel
      if side == 'right':
        x0 = self.screen_size[0] - total_width - margin
      elif side == 'center':
        x0 = self.screen_size[0] // 2 - total_width // 2
      else:  # left
        x0 = margin

      y0 = 30 + margin

      # Group title
      if group_name:
        self.draw_text(group_name, x0, y0, (0, 0, 0))

      for row, idx in enumerate(indices):
        if idx >= self.num_skeletons:
          break

        label = self.legends[idx] or ""
        color = self.skeleton_colors[idx % len(self.skeleton_colors)]
        y     = y0 + title_offset + row * spacing_y

        # Colored stripe indicating the skeleton's rendering color
        glColor3f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x0,              y)
        glVertex2f(x0 + line_length, y)
        glVertex2f(x0 + line_length, y + line_height)
        glVertex2f(x0,              y + line_height)
        glEnd()

        self.draw_text(label, x0 + line_length + 8, y - 4, (0, 0, 0))

        # ON/OFF toggle button
        button_x    = x0 + line_length + 8 + max_text_width + button_margin
        button_rect = pygame.Rect(button_x, y - 2, self.legend_button_width, self.legend_button_height)
        self.legend_buttons.append((button_rect, idx))

        if self.skeleton_visibility[idx]:
          button_color = (0.2, 0.8, 0.2)
          button_text  = "ON"
        else:
          button_color = (0.8, 0.2, 0.2)
          button_text  = "OFF"

        glColor3f(*button_color)
        glBegin(GL_QUADS)
        glVertex2f(button_rect.x,                   button_rect.y)
        glVertex2f(button_rect.x + button_rect.width, button_rect.y)
        glVertex2f(button_rect.x + button_rect.width, button_rect.y + button_rect.height)
        glVertex2f(button_rect.x,                   button_rect.y + button_rect.height)
        glEnd()

        # Button border
        glColor3f(1, 1, 1)
        glBegin(GL_LINE_LOOP)
        glVertex2f(button_rect.x,                   button_rect.y)
        glVertex2f(button_rect.x + button_rect.width, button_rect.y)
        glVertex2f(button_rect.x + button_rect.width, button_rect.y + button_rect.height)
        glVertex2f(button_rect.x,                   button_rect.y + button_rect.height)
        glEnd()

        self.draw_text(button_text, button_rect.x + 2, button_rect.y + 2, (1, 1, 1))


  def draw_button(self, rect, text, color):
    """
    Draw a filled rectangular button with a white border and centered text label.

    Parameters
    ----------
    rect : pygame.Rect
        Position and size of the button.
    text : str
        Label to display on the button.
    color : tuple of (r, g, b)
        Fill color with components in [0, 1].
    """
    glColor3f(*color)
    glBegin(GL_QUADS)
    glVertex2f(rect.x,              rect.y)
    glVertex2f(rect.x + rect.width, rect.y)
    glVertex2f(rect.x + rect.width, rect.y + rect.height)
    glVertex2f(rect.x,              rect.y + rect.height)
    glEnd()

    glColor3f(1, 1, 1)
    glBegin(GL_LINE_LOOP)
    glVertex2f(rect.x,              rect.y)
    glVertex2f(rect.x + rect.width, rect.y)
    glVertex2f(rect.x + rect.width, rect.y + rect.height)
    glVertex2f(rect.x,              rect.y + rect.height)
    glEnd()

    bold_font = pygame.font.Font(None, 26)
    bold_font.set_bold(True)
    text_surface = bold_font.render(text, True, (255, 255, 255))
    text_x = rect.x + (rect.width  - text_surface.get_width())  / 2
    text_y = rect.y + (rect.height - text_surface.get_height()) / 2
    self.draw_text(text, text_x, text_y, (1, 1, 1))


  def draw_input_field(self):
    """
    Draw the frame number text input field.

    The field shows the current frame number when inactive, or the partially
    typed number while the user is entering a value. Clicking the field activates
    it; pressing Enter confirms, Escape cancels.
    """
    glColor3f(0.2, 0.2, 0.3) if self.input_active else glColor3f(0.3, 0.3, 0.3)

    glBegin(GL_QUADS)
    glVertex2f(self.input_field.x,                    self.input_field.y)
    glVertex2f(self.input_field.x + self.input_field.width, self.input_field.y)
    glVertex2f(self.input_field.x + self.input_field.width, self.input_field.y + self.input_field.height)
    glVertex2f(self.input_field.x,                    self.input_field.y + self.input_field.height)
    glEnd()

    glColor3f(1, 1, 1)
    glBegin(GL_LINE_LOOP)
    glVertex2f(self.input_field.x,                    self.input_field.y)
    glVertex2f(self.input_field.x + self.input_field.width, self.input_field.y)
    glVertex2f(self.input_field.x + self.input_field.width, self.input_field.y + self.input_field.height)
    glVertex2f(self.input_field.x,                    self.input_field.y + self.input_field.height)
    glEnd()

    display_text = self.frame_input if self.frame_input else str(self.frame)
    self.draw_text(display_text, self.input_field.x + 5, self.input_field.y + 5, (1, 1, 1))


  def draw_slider(self):
    """
    Draw the frame scrubbing slider bar with a circular handle.

    The handle position reflects the current frame relative to the total
    frame count. Dragging the handle updates the current frame in process_event().
    """
    # Slider track
    glColor3f(0.4, 0.4, 0.4)
    glBegin(GL_QUADS)
    glVertex2f(self.slider_rect.x,                    self.slider_rect.y)
    glVertex2f(self.slider_rect.x + self.slider_rect.width, self.slider_rect.y)
    glVertex2f(self.slider_rect.x + self.slider_rect.width, self.slider_rect.y + self.slider_rect.height)
    glVertex2f(self.slider_rect.x,                    self.slider_rect.y + self.slider_rect.height)
    glEnd()

    max_frames = max((len(m) for m in self.motions if m is not None), default=0)

    handle_x = (
      self.slider_x_start + (self.frame / (max_frames - 1)) * self.slider_width
      if max_frames > 1 else self.slider_x_start
    )

    # Circular handle
    glColor3f(0.8, 0.8, 0.8)
    glBegin(GL_TRIANGLE_FAN)
    for i in range(32):
      angle = 2 * np.pi * i / 32
      x = handle_x + self.slider_handle_radius * np.cos(angle)
      y = self.slider_y + self.slider_height // 2 + self.slider_handle_radius * np.sin(angle)
      glVertex2f(x, y)
    glEnd()


  def draw_text(self, text, x, y, color):
    """
    Render a text string at a given 2D screen position using OpenGL pixel drawing.

    Must be called while in 2D projection mode (between switch_to_2d / switch_to_3d).

    Parameters
    ----------
    text : str
        Text to render. Empty strings are silently ignored.
    x, y : float
        Top-left position in screen coordinates.
    color : tuple of (r, g, b)
        Text color with components in [0, 1].
    """
    if not text:
      return

    text_color   = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    text_surface = self.font.render(text, True, text_color)
    text_data    = pygame.image.tostring(text_surface, "RGBA", True)

    glRasterPos2f(x, y + text_surface.get_height())
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)


  def process_event(self):
    """
    Process all pending pygame events and continuous keyboard input.

    Handles:
    - Window close and quit button
    - Spacebar to toggle play/pause
    - Enter to reset camera, arrow keys and WASD/QE for camera movement
    - Mouse drag for camera rotation
    - Slider drag for frame scrubbing
    - Legend button clicks to toggle skeleton visibility
    - Frame number input field
    """
    max_frames = max((len(m) for m in self.motions if m is not None), default=0)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self.done = True

      elif event.type == pygame.KEYDOWN:
        if self.input_active:
          if event.key == pygame.K_RETURN:
            if self.frame_input:
              try:
                self.frame = self.clamp_frame(int(self.frame_input))
              except ValueError:
                pass
            self.frame_input  = ""
            self.input_active = False
          elif event.key == pygame.K_BACKSPACE:
            self.frame_input = self.frame_input[:-1]
          elif event.unicode.isdigit():
            self.frame_input += event.unicode
          elif event.key == pygame.K_ESCAPE:
            self.frame_input  = ""
            self.input_active = False
        else:
          if event.key == pygame.K_RETURN:
            # Reset camera to default position and orientation
            self.translate  = self.default_translate
            self.global_rx  = 0
            self.global_ry  = 0
          elif event.key == pygame.K_SPACE:
            self.playing = not self.playing

      elif event.type == pygame.MOUSEBUTTONDOWN:
        mouse_pos = event.pos

        # Check legend toggle buttons first
        legend_button_clicked = False
        for button_rect, skeleton_idx in self.legend_buttons:
          if self.check_button_click(mouse_pos, button_rect):
            self.skeleton_visibility[skeleton_idx] = not self.skeleton_visibility[skeleton_idx]
            legend_button_clicked = True
            break

        if legend_button_clicked:
          pass
        elif self.check_button_click(mouse_pos, self.button_prev):
          self.frame = (self.frame - 1) % max_frames
        elif self.check_button_click(mouse_pos, self.button_play):
          self.playing = not self.playing
        elif self.check_button_click(mouse_pos, self.button_next):
          self.frame = (self.frame + 1) % max_frames
        elif self.check_button_click(mouse_pos, self.input_field):
          self.input_active = True
          self.frame_input  = ""
        elif self.check_button_click(mouse_pos, self.slider_rect):
          self.slider_dragging = True
          self.frame = self.get_slider_frame_from_pos(mouse_pos[0])
        elif self.check_button_click(mouse_pos, self.button_quit):
          self.done = True
        elif event.button == 1:
          self.rotate_dragging = True
        else:
          self.translate_dragging = True

        self.old_x, self.old_y = event.pos

      elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
          self.rotate_dragging = False
          self.slider_dragging  = False
        else:
          self.translate_dragging = False

      elif event.type == pygame.MOUSEMOTION:
        if self.slider_dragging:
          self.frame = self.get_slider_frame_from_pos(event.pos[0])
        elif self.rotate_dragging:
          new_x, new_y     = event.pos
          self.global_ry  -= (new_x - self.old_x) / self.screen_size[0] * np.pi
          self.global_rx  -= (new_y - self.old_y) / self.screen_size[1] * np.pi
          self.old_x, self.old_y = new_x, new_y

    # Continuous keyboard input for smooth camera movement
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_DOWN]:    self.global_rx -= self.speed_rx
    if pressed[pygame.K_UP]:      self.global_rx += self.speed_rx
    if pressed[pygame.K_LEFT]:    self.global_ry += self.speed_ry
    if pressed[pygame.K_RIGHT]:   self.global_ry -= self.speed_ry
    if pressed[pygame.K_a]:       self.translate[0] -= self.speed_trans
    if pressed[pygame.K_d]:       self.translate[0] += self.speed_trans
    if pressed[pygame.K_w]:       self.translate[1] += self.speed_trans
    if pressed[pygame.K_s]:       self.translate[1] -= self.speed_trans
    if pressed[pygame.K_q]:       self.translate[2] += self.speed_zoom
    if pressed[pygame.K_e]:       self.translate[2] -= self.speed_zoom
    if pressed[pygame.K_COMMA]:   self.frame = (self.frame - 1) % max_frames
    if pressed[pygame.K_PERIOD]:  self.frame = (self.frame + 1) % max_frames

    # Update combined rotation matrix from current Euler angles
    grx = euler.euler2mat(self.global_rx, 0, 0)
    gry = euler.euler2mat(0, self.global_ry, 0)
    self.rotation_R = grx.dot(gry)


  def set_joints(self, joints):
    """
    Replace the current joint sets with new ones.

    Parameters
    ----------
    joints : dict or sequence of dict
        New skeleton(s) to use. Count must match the existing motion sequences.
    """
    if isinstance(joints, dict):
      self.joints = (joints,)
    else:
      self.joints = tuple(joints)
    self.num_skeletons = len(self.joints)

    if self.motions and len(self.motions) != self.num_skeletons:
      raise ValueError(
        f"Number of joint sets ({self.num_skeletons}) must match number of motion sequences ({len(self.motions)})."
      )


  def calculate_root_offset(self):
    """
    Record the root joint world position from the first frame of each skeleton.

    Used as a reference offset so skeletons can optionally be centered or
    compared relative to their starting position.

    Returns
    -------
    list of np.ndarray or None
        One (3,) position array per skeleton, or None if no data is available.
    """
    if not self.joints or not self.motions:
      return None

    positions = []
    for joints, motions in zip(self.joints, self.motions):
      if joints and motions and len(motions) > 0:
        joints['root'].set_motion(motions[0])
        root_coord = np.squeeze(joints['root'].coordinate).copy()
        positions.append(root_coord)

    return positions if positions else None


  def draw(self):
    """
    Render one frame: clear buffers, draw the floor, then draw all visible skeletons.

    For each visible skeleton, set_motion() is called with the current frame to
    update all joint coordinates via forward kinematics. Joints are drawn as
    points and bones as lines, both in the skeleton's assigned color.
    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    self.draw_floor()
    glDisable(GL_LIGHTING)

    for skeleton_idx, (joints, motions) in enumerate(zip(self.joints, self.motions)):
      if not self.skeleton_visibility[skeleton_idx]:
        continue

      if joints is None or motions is None:
        continue

      color = self.skeleton_colors[skeleton_idx % len(self.skeleton_colors)]

      if self.frame < len(motions):
        joints['root'].set_motion(motions[self.frame])

      # Draw joint positions as points
      glBegin(GL_POINTS)
      glColor3f(*color)
      for j in joints.values():
        joint_pos   = np.squeeze(j.coordinate)
        rotated_pos = joint_pos.dot(self.rotation_R) + self.translate
        glVertex3f(*rotated_pos.astype(np.float32))
      glEnd()

      # Draw bones as lines between each joint and its parent
      glBegin(GL_LINES)
      glColor3f(*color)
      for j in joints.values():
        if j.parent is not None:
          child_pos  = np.squeeze(j.coordinate)
          parent_pos = np.squeeze(j.parent.coordinate)
          coord_x = child_pos.dot(self.rotation_R)  + self.translate
          coord_y = parent_pos.dot(self.rotation_R) + self.translate
          glVertex3f(*coord_x.astype(np.float32))
          glVertex3f(*coord_y.astype(np.float32))
      glEnd()

    self.draw_ui()


  def set_motion(self, motions):
    """
    Replace the current motion sequences with new ones.

    Parameters
    ----------
    motions : list or sequence of list
        New motion sequence(s). Count must match the number of skeletons.
    """
    if isinstance(motions, list):
      self.motions = (motions,)
    else:
      self.motions = tuple(motions)

    if len(self.motions) != self.num_skeletons:
      raise ValueError(
        f"Number of motion sequences ({len(self.motions)}) must match number of skeletons ({self.num_skeletons})."
      )

    self.initial_root_positions = self.calculate_root_offset()


  def run(self):
    """
    Start the main render loop.

    Runs until the user closes the window or clicks Quit. Each iteration
    processes events, advances the frame if playing, renders the scene,
    updates the window caption, and caps the frame rate at self.fps.
    """
    print("Viewer num_skeletons:", self.num_skeletons)
    print("Viewer joints:",        len(self.joints))
    print("Viewer motions:",       len(self.motions))

    while not self.done:
      self.process_event()

      if self.playing:
        max_frames  = max((len(m) for m in self.motions if m is not None), default=0)
        self.frame += 1
        if self.frame >= max_frames:
          self.frame = 0

      self.draw()

      max_frames = max((len(m) for m in self.motions if m is not None), default=0)
      pygame.display.set_caption(
        'AMC Parser - frame %d / %d (%d skeletons)' % (self.frame, max_frames, self.num_skeletons)
      )
      pygame.display.flip()
      self.clock.tick(self.fps)

    pygame.quit()


if __name__ == '__main__':
  from motion_player.amc_parser import parse_asf, parse_amc
  asf_path = './data/01/01.asf'
  amc_path = './data/01/01_01.amc'
  joints  = parse_asf(asf_path)
  motions = parse_amc(amc_path)
  v = Viewer(joints, motions)
  v.run()