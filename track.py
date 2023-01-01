import pygame
from utils import scale_image


FINISH_LINE_POSITION = (130, 250)
class Track:
    def __init__(self, screen):
        self.screen = screen

        self.grass = scale_image(pygame.image.load("img/grass.jpg"), 2.5)
        self.track = scale_image(pygame.image.load("img/track.png"), 0.9)

        self.track_border = scale_image(pygame.image.load("img/track-border.png"), 0.9)
        self.track_border_mask = pygame.mask.from_surface(self.track_border)

        self.finish_line = pygame.image.load("img/finish.png")
        self.finish_line_mask = pygame.mask.from_surface(self.finish_line)

    def draw(self):
        images = [
            (self.grass, (0, 0)),
            (self.track, (0, 0)),
            (self.track_border, (0, 0)),
            (self.finish_line, FINISH_LINE_POSITION)
        ]

        for img, pos in images:
            self.screen.blit(img, pos)
