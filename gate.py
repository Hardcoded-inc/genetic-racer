import pygame
from utils import scale_image

GATES_AMOUNT = 11


class Gate:
    def __init__(self, screen, car, prev_gate=None):
        if prev_gate is None or prev_gate.index == GATES_AMOUNT:
            self.index = 1
        else:
            self.index = prev_gate.index + 1

        self.screen = screen

        self.gate = scale_image(pygame.image.load("img/gates/{}.png".format(self.index)), 0.9)
        self.gate_mask = pygame.mask.from_surface(self.gate)

        car.update_currect_gate(self)

    def draw(self):
        self.screen.blit(self.gate, (0, 0))
