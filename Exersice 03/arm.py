import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Segment:
    """Represents a single segment of the robotic arm."""
    
    def __init__(self, length):
        """
        Initialize a segment with given length.
        
        Args:
            length: length of the segment along Y axis
        """
        self.length = length
        self._vertices = None
        self._faces = None
        self._create_vertices()
        
    def _create_vertices(self):
        """Create the vertices for the segment."""
        self._vertices = np.array([
            [0, self.length, 0],      # 0: tip
            [-1, 0.0, 1],             # 1: base corner
            [-1, 0.0, -1],            # 2: base corner
            [1, 0.0, -1],             # 3: base corner
            [1, 0.0, 1]               # 4: base corner
        ], dtype=float)
    
    def _create_faces(self):
        """Create face indices."""
        self._faces = np.array([
            [0, 2, 1],  # Front-left face
            [0, 1, 4],  # Front-right face
            [0, 4, 3],  # Back-right face
            [0, 3, 2],  # Back-left face
            [1, 2, 3],  # Bottom face (part 1)
            [1, 3, 4]   # Bottom face (part 2)
        ])
    
    @property
    def vertices(self):
        """Get the vertices of the segment."""
        return self._vertices.copy()
    
    @vertices.setter
    def vertices(self, new_vertices):
        """Set new vertices for the segment."""
        self._vertices = new_vertices.copy()
    
    @property
    def faces(self):
        """Get the face indices."""
        if self._faces is None:
            self._create_faces()
        return self._faces
    
    @property
    def triangles(self):
        """Get triangles as (N, 3, 3) array."""
        if self._faces is None:
            self._create_faces()
        return self._vertices[self._faces]


class Arm:
    """Represents a 3-link robotic arm with visualization."""
    
    def __init__(self, ax, l1, l2, l3):
        """
        Initialize a 3-link robotic arm.
        
        Args:
            ax: matplotlib 3D axis object
            l1, l2, l3: lengths of the three segments
        """
        self.ax = ax
        self.lengths = [l1, l2, l3]
        
        # Create segments
        self.segments = [Segment(l1), Segment(l2), Segment(l3)]
        
        # Store patches for rendering
        self.patches = [None, None, None]
        
        # Color map for segments
        self.color_map = np.array([
            [0, 1, 0],
            [0, 0.75, 0.25],
            [0, 0.5, 0.5],
            [0, 0.25, 0.75],
            [0, 0, 1]
        ])
        
        # Initial setup: position segments relative to each other
        self._setup_segments()
    
    def _setup_segments(self):
        """
        Setup segments so they connect:
        - Segment 0: starts at origin
        - Segment 1: starts where segment 0 ends
        - Segment 2: starts where segment 1 ends
        """
        # Segment 0 stays at origin
        
        # Segment 1: translate by l1 (length of segment 0) along Y
        v1 = self.segments[1].vertices
        v1[:, 1] += self.lengths[0]  # Move all Y coordinates by l1
        self.segments[1].vertices = v1
        
        # Segment 2: translate by l1 + l2 along Y
        v2 = self.segments[2].vertices
        v2[:, 1] += self.lengths[0] + self.lengths[1]  # Move all Y coordinates by l1 + l2
        self.segments[2].vertices = v2
    
    def get_vertices(self, segment_idx):
        """
        Get vertices of a specific segment.
        
        Args:
            segment_idx: index of segment (0, 1, or 2)
        
        Returns:
            vertices: (5, 3) array of vertices
        """
        if 0 <= segment_idx < 3:
            return self.segments[segment_idx].vertices
        else:
            raise ValueError("segment_idx must be 0, 1, or 2")
    
    def set_vertices(self, segment_idx, new_vertices):
        """
        Set new vertices for a segment.
        
        Args:
            segment_idx: index of segment (0, 1, or 2)
            new_vertices: (5, 3) array of new vertices
        """
        if 0 <= segment_idx < 3:
            self.segments[segment_idx].vertices = new_vertices
        else:
            raise ValueError("segment_idx must be 0, 1, or 2")
    
    def _create_patch(self, segment_idx):
        """
        Create a Poly3DCollection patch from segment.
        
        Args:
            segment_idx: index of the segment
        
        Returns:
            patch: Poly3DCollection object
        """
        segment = self.segments[segment_idx]
        triangles = segment.triangles
        
        patch = Poly3DCollection(triangles,
                                alpha=0.6,
                                edgecolor='darkslategray',
                                linewidths=2.0)
        
        # Set face colors
        face_colors = []
        for face in segment.faces:
            face_colors.append(self.color_map[face].mean(axis=0))
        patch.set_facecolor(face_colors)
        
        self.ax.add_collection3d(patch)
        return patch
    
    def _remove_patches(self):
        """Remove all current patches from the axis."""
        for patch in self.patches:
            if patch is not None:
                patch.remove()
        self.patches = [None, None, None]
    
    def plot(self):
        """Render all segments to the 3D plot."""
        self._remove_patches()
        
        for i in range(3):
            self.patches[i] = self._create_patch(i)