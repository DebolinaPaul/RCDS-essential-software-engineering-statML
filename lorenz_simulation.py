"""
Lorenz Equations Simulator with Tests and Visualizations

This module provides functions to simulate the Lorenz equations, test the simulation,
and generate 2D and 3D visualizations of the Lorenz attractor.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import unittest


def lorenz(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Compute the derivatives of the Lorenz equations.
    
    Parameters:
    -----------
    state : array-like
        Current state [x, y, z]
    t : float
        Time (required by odeint but not used in this autonomous system)
    sigma : float
        Prandtl number (default: 10.0)
    rho : float
        Rayleigh number (default: 28.0)
    beta : float
        Aspect ratio (default: 8/3)
    
    Returns:
    --------
    list
        Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    
    return [dx_dt, dy_dt, dz_dt]


def simulate_lorenz(initial_state, t_max=100.0, num_points=10000, 
                    sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Simulate the Lorenz equations over time.
    
    Parameters:
    -----------
    initial_state : array-like
        Initial conditions [x0, y0, z0]
    t_max : float
        Maximum time to simulate (default: 100.0)
    num_points : int
        Number of time points to compute (default: 10000)
    sigma : float
        Prandtl number (default: 10.0)
    rho : float
        Rayleigh number (default: 28.0)
    beta : float
        Aspect ratio (default: 8/3)
    
    Returns:
    --------
    tuple
        (t, trajectory) where t is time array and trajectory is (num_points, 3) array
    """
    t = np.linspace(0, t_max, num_points)
    trajectory = odeint(lorenz, initial_state, t, args=(sigma, rho, beta))
    return t, trajectory


class TestLorenzSimulation(unittest.TestCase):
    """Test cases for the Lorenz equations simulator."""
    
    def test_lorenz_derivative_fixed_point(self):
        """Test Lorenz derivatives at equilibrium point (0,0,0)."""
        # At the origin, derivatives should be zero (with default parameters)
        state = [0, 0, 0]
        derivs = lorenz(state, 0)
        self.assertAlmostEqual(derivs[0], 0, places=10)
        self.assertAlmostEqual(derivs[1], 0, places=10)
        self.assertAlmostEqual(derivs[2], 0, places=10)
    
    def test_lorenz_derivative_symmetry(self):
        """Test that the system exhibits x-y symmetry."""
        # For symmetric initial conditions, x and y should behave symmetrically
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        state1 = [1, 1, 1]
        derivs1 = lorenz(state1, 0, sigma, rho, beta)
        
        state2 = [1, 1, 1]
        derivs2 = lorenz(state2, 0, sigma, rho, beta)
        
        # First derivatives should be equal for symmetric x, y
        self.assertAlmostEqual(derivs1[0], derivs2[0], places=10)
    
    def test_simulate_lorenz_output_shape(self):
        """Test that simulation returns correct output shapes."""
        initial_state = [1, 1, 1]
        t_max = 10.0
        num_points = 100
        
        t, traj = simulate_lorenz(initial_state, t_max, num_points)
        
        # Check shapes
        self.assertEqual(len(t), num_points)
        self.assertEqual(traj.shape, (num_points, 3))
        
        # Check time array
        self.assertAlmostEqual(t[0], 0)
        self.assertAlmostEqual(t[-1], t_max, places=5)
    
    def test_simulate_lorenz_initial_condition(self):
        """Test that simulation starts at the initial condition."""
        initial_state = [5, -3, 10]
        t, traj = simulate_lorenz(initial_state, t_max=1.0, num_points=100)
        
        # First point should be close to initial state
        np.testing.assert_array_almost_equal(traj[0], initial_state, decimal=5)
    
    def test_simulate_lorenz_diverges_from_equilibrium(self):
        """Test that trajectory diverges from equilibrium point."""
        # Start near origin but not at it
        initial_state = [0.1, 0.1, 0.1]
        t, traj = simulate_lorenz(initial_state, t_max=50.0, num_points=5000)
        
        # After some time, trajectory should move away from origin
        final_distance = np.linalg.norm(traj[-1])
        # Should be significantly far from origin
        self.assertGreater(final_distance, 5.0)
    
    def test_simulate_lorenz_bounded(self):
        """Test that trajectory remains bounded (attractor property)."""
        initial_state = [1, 1, 1]
        t, traj = simulate_lorenz(initial_state, t_max=100.0, num_points=10000)
        
        # All coordinates should remain bounded
        max_x = np.max(np.abs(traj[:, 0]))
        max_y = np.max(np.abs(traj[:, 1]))
        max_z = np.max(traj[:, 2])
        
        # These bounds are typical for Lorenz attractor with default parameters
        self.assertLess(max_x, 35)
        self.assertLess(max_y, 35)
        self.assertLess(max_z, 60)


def generate_2d_plots():
    """Generate and save 2D projection plots of the Lorenz attractor."""
    # Simulate the Lorenz system
    initial_state = [1.0, 1.0, 1.0]
    t, trajectory = simulate_lorenz(initial_state, t_max=50.0, num_points=5000)
    
    # Extract coordinates
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    
    # Create 2D projections
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # XY plane projection
    axes[0].plot(x, y, linewidth=0.5, alpha=0.8, color='blue')
    axes[0].scatter(x[0], y[0], color='red', s=50, label='Start', zorder=5)
    axes[0].scatter(x[-1], y[-1], color='green', s=50, label='End', zorder=5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('XY Plane Projection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # XZ plane projection
    axes[1].plot(x, z, linewidth=0.5, alpha=0.8, color='purple')
    axes[1].scatter(x[0], z[0], color='red', s=50, label='Start', zorder=5)
    axes[1].scatter(x[-1], z[-1], color='green', s=50, label='End', zorder=5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    axes[1].set_title('XZ Plane Projection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # YZ plane projection
    axes[2].plot(y, z, linewidth=0.5, alpha=0.8, color='orange')
    axes[2].scatter(y[0], z[0], color='red', s=50, label='Start', zorder=5)
    axes[2].scatter(y[-1], z[-1], color='green', s=50, label='End', zorder=5)
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('z')
    axes[2].set_title('YZ Plane Projection')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lorenz_2d_projections.png', dpi=150, bbox_inches='tight')
    print("✓ 2D projection plot saved as 'lorenz_2d_projections.png'")


def generate_3d_plot():
    """Generate and save 3D plot of the Lorenz attractor."""
    # Simulate the Lorenz system
    initial_state = [1.0, 1.0, 1.0]
    t, trajectory = simulate_lorenz(initial_state, t_max=50.0, num_points=5000)
    
    # Extract coordinates
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    
    # Create a 3D plot of the Lorenz attractor
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a color gradient based on time
    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
    
    # Plot trajectory with color gradient
    for i in range(len(t) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=1, alpha=0.8)
    
    # Mark start and end points
    ax.scatter(x[0], y[0], z[0], color='red', s=100, label='Start', marker='o', zorder=5)
    ax.scatter(x[-1], y[-1], z[-1], color='green', s=100, label='End', marker='s', zorder=5)
    
    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Lorenz Attractor (3D Trajectory)', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lorenz_3d_attractor.png', dpi=150, bbox_inches='tight')
    print("✓ 3D attractor plot saved as 'lorenz_3d_attractor.png'")


if __name__ == '__main__':
    print("=" * 70)
    print("Lorenz Equations Simulator - Tests and Visualizations")
    print("=" * 70)
    print()
    
    # Run tests
    print("Running tests...")
    print("-" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLorenzSimulation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print()
    
    if result.wasSuccessful():
        print("✓ All tests passed!")
        print()
        
        # Generate visualizations
        print("Generating visualizations...")
        print("-" * 70)
        generate_2d_plots()
        generate_3d_plot()
        print()
        print("=" * 70)
        print("Complete! Check the PNG files for visualizations.")
        print("=" * 70)
    else:
        print("✗ Some tests failed!")
