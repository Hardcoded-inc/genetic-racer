import pygame
from utils import scale_image, flip_surface


GATES_AMOUNT = 27
RAY_LEN = 560

REWARD_VAL = 10

class Gate:
    def __init__(self, screen, car, prev_gate=None):
        self.reward_val = REWARD_VAL

        if prev_gate is None or prev_gate.index == GATES_AMOUNT:
            self.index = 1
        else:
            self.index = prev_gate.index + 1

        self.screen = screen

        self.gate = scale_image(pygame.image.load("img/gates/{}.png".format(self.index)), 0.9)
        self.mask = pygame.mask.from_surface(self.gate)

        self.beam_surface = pygame.Surface((RAY_LEN, RAY_LEN), pygame.SRCALPHA)
        self.flipped_masks = flip_surface(self.gate)

    def draw(self):
        self.screen.blit(self.gate, (0, 0))
