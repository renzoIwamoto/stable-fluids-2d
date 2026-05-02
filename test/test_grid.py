import numpy as np

from src.grid import FluidGrid

def test_grid_init():
    N = 5
    grid = FluidGrid(5)

    expected_shape = (N+2, N+2)
    assert grid.u.shape == expected_shape
    assert grid.v.shape == expected_shape
    assert grid.density.shape == expected_shape

    assert np.all(grid.u == 0)
    assert np.all(grid.v == 0)
    assert np.all (grid.density == 0)

def test_grid_reset():
    """Despues de modificar y resetear, los campos vuelven a cero."""
    grid = FluidGrid(N=32)

    grid.u[10, 10] = 5.0
    grid.density[5, 5] = 1.0

    grid.reset()

    assert np.all(grid.u == 0)
    assert np.all(grid.density == 0)
