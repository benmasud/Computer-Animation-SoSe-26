import numpy as np
import pytest
from typing import Union

# Assuming the functions from the notebook are available
# Import them or define them in the same notebook

class TestSlerpQuaternion:
    """Test suite for spherical linear interpolation of quaternions"""
    
    def test_slerp_scalar_t_at_start(self):
        """SLERP at t=0 should return start quaternion"""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = axis_angle_quaternion("Z", np.pi / 2)
        
        result = slerp_quaternion(q1, q2, 0.0)
        assert np.allclose(result, q1), "SLERP at t=0 should equal q_start"
    
    def test_slerp_scalar_t_at_end(self):
        """SLERP at t=1 should return end quaternion"""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = axis_angle_quaternion("Z", np.pi / 2)
        
        result = slerp_quaternion(q1, q2, 1.0)
        assert np.allclose(result, q2), "SLERP at t=1 should equal q_end"
    
    def test_slerp_midpoint_constant_velocity(self):
        """SLERP midpoint should be equidistant in rotation angle"""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = axis_angle_quaternion("Z", np.pi / 2)
        
        q_mid = slerp_quaternion(q1, q2, 0.5)
        
        # Calculate angles
        angle_start_to_mid = relative_rotation_angle(
            quaternion_to_rotation_matrix(q1),
            quaternion_to_rotation_matrix(q_mid)
        )
        angle_mid_to_end = relative_rotation_angle(
            quaternion_to_rotation_matrix(q_mid),
            quaternion_to_rotation_matrix(q2)
        )
        
        # Angles should be equal (both ~45 degrees)
        assert np.isclose(angle_start_to_mid, angle_mid_to_end, atol=1e-6), \
            "Midpoint should be equidistant in rotation"
    
    def test_slerp_array_t(self):
        """SLERP with array t should return correct shape"""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = axis_angle_quaternion("Z", np.pi / 2)
        
        t_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = slerp_quaternion(q1, q2, t_values)
        
        assert result.shape == (5, 4), "Result shape should be (5, 4)"
        assert np.allclose(result[0], q1), "First should be q_start"
        assert np.allclose(result[-1], q2), "Last should be q_end"
    
    def test_slerp_sign_ambiguity_shortest_path(self):
        """SLERP should take shortest path with opposite-signed quaternions"""
        q_a = np.array([0.7071, 0.7071, 0.0, 0.0])
        q_b = np.array([-0.7071, -0.7071, 0.0, 0.0])  # Same rotation, opposite sign
        
        result = slerp_quaternion(q_a, q_b, 0.5)
        
        # Result should be close to one of the inputs (short path)
        dist_to_a = np.linalg.norm(result - q_a)
        dist_to_b = np.linalg.norm(result - q_b)
        
        # One distance should be small (short path taken)
        assert np.min([dist_to_a, dist_to_b]) < 0.5, \
            "SLERP should take the shortest path"
    
    def test_slerp_constant_angular_velocity(self):
        """SLERP should maintain approximately constant angular velocity"""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = axis_angle_quaternion("Z", np.pi / 2)
        
        t_values = np.linspace(0, 1, 11)
        q_slerp = slerp_quaternion(q1, q2, t_values)
        
        # Calculate consecutive rotation angles
        angles = []
        for i in range(len(q_slerp) - 1):
            angle = relative_rotation_angle(
                quaternion_to_rotation_matrix(q_slerp[i]),
                quaternion_to_rotation_matrix(q_slerp[i + 1])
            )
            angles.append(angle)
        
        angles = np.array(angles)
        std_dev = np.std(angles)
        
        # Standard deviation should be very small for constant velocity
        assert std_dev < 1e-6, \
            f"Angular velocity should be constant (std={std_dev})"
    
    def test_slerp_small_angles_no_numerical_issues(self):
        """SLERP should handle very small rotation angles correctly"""
        q_start = np.array([1.0, 0.0, 0.0, 0.0])
        q_end = axis_angle_quaternion("X", 0.001)  # Very small angle
        
        result = slerp_quaternion(q_start, q_end, 0.5)
        
        # Result should be a valid normalized quaternion
        norm = np.linalg.norm(result)
        assert np.isclose(norm, 1.0), "Result should be normalized"
    
    def test_slerp_output_normalized(self):
        """All output quaternions should be normalized"""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = axis_angle_quaternion("Z", np.pi / 2)
        
        t_values = np.linspace(0, 1, 20)
        result = slerp_quaternion(q1, q2, t_values)
        
        norms = np.linalg.norm(result, axis=-1)
        assert np.allclose(norms, 1.0), "All quaternions should be normalized"
    
    def test_slerp_scalar_t_returns_scalar(self):
        """SLERP with scalar t should return 1D quaternion"""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = axis_angle_quaternion("Z", np.pi / 2)
        
        result = slerp_quaternion(q1, q2, 0.5)
        
        assert result.shape == (4,), "Scalar t should return 1D quaternion"


class TestLerpEuler:
    """Test suite for linear interpolation of Euler angles"""
    
    def test_lerp_scalar_t_at_start(self):
        """LERP at t=0 should return start angles"""
        euler_start = np.array([0.0, 0.0, 0.0])
        euler_end = np.array([np.pi/4, np.pi/6, np.pi/3])
        
        result = lerp_euler(euler_start, euler_end, 0.0)
        assert np.allclose(result, euler_start), "LERP at t=0 should equal euler_start"
    
    def test_lerp_scalar_t_at_end(self):
        """LERP at t=1 should return end angles"""
        euler_start = np.array([0.0, 0.0, 0.0])
        euler_end = np.array([np.pi/4, np.pi/6, np.pi/3])
        
        result = lerp_euler(euler_start, euler_end, 1.0)
        assert np.allclose(result, euler_end), "LERP at t=1 should equal euler_end"
    
    def test_lerp_midpoint_is_average(self):
        """LERP midpoint should be the average of start and end"""
        euler_start = np.array([0.0, 0.0, 0.0])
        euler_end = np.array([np.pi/4, np.pi/6, np.pi/3])
        
        result = lerp_euler(euler_start, euler_end, 0.5)
        expected = (euler_start + euler_end) / 2
        
        assert np.allclose(result, expected), "Midpoint should be average"
    
    def test_lerp_array_t(self):
        """LERP with array t should return correct shape"""
        euler_start = np.array([0.0, 0.0, 0.0])
        euler_end = np.array([np.pi/4, np.pi/6, np.pi/3])
        
        t_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = lerp_euler(euler_start, euler_end, t_values)
        
        assert result.shape == (5, 3), "Result shape should be (5, 3)"
        assert np.allclose(result[0], euler_start), "First should be euler_start"
        assert np.allclose(result[-1], euler_end), "Last should be euler_end"
    
    def test_lerp_linearity(self):
        """LERP should produce linear progression"""
        euler_start = np.array([0.0, 0.0, 0.0])
        euler_end = np.array([1.0, 2.0, 3.0])
        
        t_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = lerp_euler(euler_start, euler_end, t_values)
        
        # Check each angle progresses linearly
        for i in range(3):  # For each angle
            expected = euler_start[i] + t_values * (euler_end[i] - euler_start[i])
            assert np.allclose(result[:, i], expected), \
                f"Angle {i} should progress linearly"
    
    def test_lerp_scalar_t_returns_scalar(self):
        """LERP with scalar t should return 1D euler angles"""
        euler_start = np.array([0.0, 0.0, 0.0])
        euler_end = np.array([np.pi/4, np.pi/6, np.pi/3])
        
        result = lerp_euler(euler_start, euler_end, 0.5)
        
        assert result.shape == (3,), "Scalar t should return 1D euler angles"
    
    def test_lerp_different_ranges(self):
        """LERP should work with different angle ranges"""
        euler_start = np.array([-np.pi, 0.0, np.pi/2])
        euler_end = np.array([np.pi, np.pi/2, np.pi])
        
        result = lerp_euler(euler_start, euler_end, 0.5)
        expected = (euler_start + euler_end) / 2
        
        assert np.allclose(result, expected), \
            "LERP should work with different angle ranges"


class TestInterpolationComparison:
    """Test comparisons between SLERP and LERP"""
    
    def test_slerp_vs_lerp_directions(self):
        """Compare directions from SLERP and LERP at various t values"""
        euler_start = np.array([0.0, 0.0, 0.0])
        euler_end = np.array([np.pi/4, np.pi/6, np.pi/3])
        
        q_start = euler_to_quaternion(euler_start, order=ORDER)
        q_end = euler_to_quaternion(euler_end, order=ORDER)
        
        t_values = np.linspace(0, 1, 5)
        
        slerp_dirs = directions_from_quaternions(
            slerp_quaternion(q_start, q_end, t_values)
        )
        lerp_eulers = lerp_euler(euler_start, euler_end, t_values)
        lerp_dirs = directions_from_eulers(lerp_eulers, order=ORDER)
        
        # Both should produce valid direction vectors
        assert np.allclose(np.linalg.norm(slerp_dirs, axis=-1), 1.0), \
            "SLERP directions should be normalized"
        assert np.allclose(np.linalg.norm(lerp_dirs, axis=-1), 1.0), \
            "LERP directions should be normalized"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])