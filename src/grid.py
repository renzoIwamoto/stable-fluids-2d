import numpy as np

class FluidGrid:

    def __init__(self, N: int):
        self.N = N
        shape = (N+2 , N+2)

        self.u = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.density = np.zeros(shape, dtype=np.float32)

        self.u_prev = np.zeros(shape, dtype=np.float32)
        self.v_prev = np.zeros(shape, dtype=np.float32)
        self.density_prev = np.zeros(shape, dtype=np.float32)

    def reset(self):
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.density.fill(0.0)
        self.u_prev.fill(0.0)
        self.v_prev.fill(0.0)
        self.density_prev.fill(0.0)

