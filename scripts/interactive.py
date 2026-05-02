"""
Demo interactiva del fluido con Stable Fluids 2D completo.

Controles:
    Boton izquierdo + mover : aplica fuerza al fluido.
    Boton derecho + mover   : inyecta tinta (density).
    Tecla R                 : reset.
    Tecla V                 : alterna visualizar density vs velocidad.
"""

import numpy as np
import pygame

from src.grid import FluidGrid
from src.render import Renderer
from src.solver import add_source, diffuse, advect, project


def main():
    # Configuracion
    N = 96
    CELL_SIZE = 6
    FPS = 60

    DT = 0.1
    VISCOSITY = 0.0001
    DIFFUSION = 0.0001
    SOLVER_ITERATIONS = 20

    FORCE_STRENGTH = 5.0
    SOURCE_STRENGTH = 100.0

    visualize_velocity = False

    grid = FluidGrid(N=N)
    renderer = Renderer(
        N=N, cell_size=CELL_SIZE,
        title="Stable Fluids 2D - Completo"
    )

    shape = (N + 2, N + 2)
    force_u = np.zeros(shape, dtype=np.float32)
    force_v = np.zeros(shape, dtype=np.float32)
    source_density = np.zeros(shape, dtype=np.float32)

    # Buffers de trabajo para project
    p_buf = np.zeros(shape, dtype=np.float32)
    div_buf = np.zeros(shape, dtype=np.float32)

    running = True
    while running:
        # 1. Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    grid.reset()
                elif event.key == pygame.K_v:
                    visualize_velocity = not visualize_velocity

        # 2. Mouse
        mouse = renderer.get_mouse_state()
        if mouse["inside"]:
            i, j = mouse["i"], mouse["j"]
            if mouse["left"]:
                force_u[i-1:i+2, j-1:j+2] += mouse["di"] * FORCE_STRENGTH
                force_v[i-1:i+2, j-1:j+2] += mouse["dj"] * FORCE_STRENGTH
            if mouse["right"]:
                source_density[i-1:i+2, j-1:j+2] += SOURCE_STRENGTH

        # === VELOCITY STEP ===
        # 3. Add forces
        add_source(grid.u, force_u, DT)
        add_source(grid.v, force_v, DT)

        # 4. Diffuse
        np.copyto(grid.u_prev, grid.u)
        np.copyto(grid.v_prev, grid.v)
        diffuse(grid.u, grid.u_prev, VISCOSITY, DT, kind="u",
                iterations=SOLVER_ITERATIONS)
        diffuse(grid.v, grid.v_prev, VISCOSITY, DT, kind="v",
                iterations=SOLVER_ITERATIONS)

        # 5. Project
        project(grid.u, grid.v, p_buf, div_buf, iterations=SOLVER_ITERATIONS)

        # 6. Advect velocidad
        np.copyto(grid.u_prev, grid.u)
        np.copyto(grid.v_prev, grid.v)
        advect(grid.u, grid.u_prev, grid.u_prev, grid.v_prev, DT, kind="u")
        advect(grid.v, grid.v_prev, grid.u_prev, grid.v_prev, DT, kind="v")

        # 7. Project (segunda vez)
        project(grid.u, grid.v, p_buf, div_buf, iterations=SOLVER_ITERATIONS)

        # === DENSITY STEP ===
        # 8. Add source
        add_source(grid.density, source_density, DT)

        # 9. Diffuse
        np.copyto(grid.density_prev, grid.density)
        diffuse(grid.density, grid.density_prev, DIFFUSION, DT,
                kind="scalar", iterations=SOLVER_ITERATIONS)

        # 10. Advect
        np.copyto(grid.density_prev, grid.density)
        advect(grid.density, grid.density_prev, grid.u, grid.v, DT,
               kind="scalar")

        # 11. Limpiar buffers de input
        force_u.fill(0.0)
        force_v.fill(0.0)
        source_density.fill(0.0)

        # 12. Renderizar
        if visualize_velocity:
            velocity_magnitude = np.sqrt(grid.u**2 + grid.v**2)
            renderer.draw(velocity_magnitude)
        else:
            renderer.draw(grid.density)

        renderer.tick(FPS)

    renderer.close()


if __name__ == "__main__":
    main()