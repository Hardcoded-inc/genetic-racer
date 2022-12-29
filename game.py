import pygame
from car import Car
from track import Track

class Game:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Set the window size and title
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Racing Game")

        # Set the background color
        self.bg_color = (0, 0, 0)

        # Set the clock to control the frame rate
        self.clock = pygame.time.Clock()

        # Create the track and car
        self.track = Track(self.screen)
        self.car = Car(self.screen)

    def run(self):
        # Main game loop
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update the game state
            self.update()

            # Draw the game
            self.draw()

            # Limit the frame rate
            self.clock.tick(60)

        # Quit Pygame
        pygame.quit()

    def update(self):
        # Update the car's position
        self.car.update()

    def draw(self):
        # Draw the track and car
        self.track.draw()
        self.car.draw()

        # Update the display
        pygame.display.flip()
