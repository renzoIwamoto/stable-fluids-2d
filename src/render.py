"""
Renderizado del fluido con Pygame.
"""

import numpy as np
import pygame


class Renderer:
    def __init__(self, N: int, cell_size: int = 8, title: str = "Stable Fluids 2D"):
        self.N = N
        self.cell_size = cell_size
        self.window_size = N * cell_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def draw(self, field: np.ndarray) -> None:
        interior = field[1:-1, 1:-1]
        clipped = np.clip(interior, 0.0, 1.0)
        gray = (clipped * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb = np.flip(rgb, axis=1)
        surface = pygame.surfarray.make_surface(rgb)
        scaled = pygame.transform.scale(
            surface, (self.window_size, self.window_size)
        )
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self) -> None:
        pygame.quit()

    def get_mouse_state(self) -> dict:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        rel_x, rel_y = pygame.mouse.get_rel()
        buttons = pygame.mouse.get_pressed()

        i = mouse_x // self.cell_size + 1
        j = (self.window_size - mouse_y) // self.cell_size + 1

        inside = (1 <= i <= self.N) and (1 <= j <= self.N)

        i = int(np.clip(i, 1, self.N))
        j = int(np.clip(j, 1, self.N))

        di = rel_x / self.cell_size
        dj = -rel_y / self.cell_size

        return {
            "i": i,
            "j": j,
            "di": di,
            "dj": dj,
            "left": buttons[0],
            "right": buttons[2],
            "inside": inside,
        }