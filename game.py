import pygame
from car import Car
from track import Track
from utils import scale_image


WIDTH, HEIGHT = (810, 810)
FPS = 30

class Game:
    actions_count = 8
    state_size = 11
    frame_rate = FPS

    def __init__(self):
        pygame.init()

        # Set the window size and title
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Genetic Racer")

        # Set the clock to control the frame rate
        self.clock = pygame.time.Clock()

        # Create the track and car
        self.track = Track(self.screen)
        self.car = Car(self.screen)

    def run(self):
        # Main game loop
        running = True
        while running:

            self.clock.tick(FPS)
            self.update()
            self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()

    def update(self):
        self.car.update()

    def draw(self):
        # Draw the track and car
        self.track.draw()
        self.car.draw()

        if self.car.collide(self.track.track_border_mask) != None:
            self.car.reset()

        # Update the display
        pygame.display.flip()
