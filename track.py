import pygame

class Track:
    def __init__(self, screen):
        self.screen = screen

        # Load the track image and get its rect
        self.image = pygame.image.load("track.png")
        self.rect = self.image.get_rect()

    def draw(self):
        self.screen.blit(self.image, self.rect)
