import pygame

class Car:
    def __init__(self, screen):
        self.screen = screen

        # Load the car image and get its rect
        self.image = pygame.image.load("img/car.png")
        self.rect = self.image.get_rect()

        # Set the car's starting position
        self.rect.center = (320, 240)

        # Set the car's speed
        self.speed = 0

        # Set the car's acceleration and deceleration
        self.acceleration = 0.1
        self.deceleration = 0.1


    def update(self):
        # Handle keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.speed += self.acceleration
        elif keys[pygame.K_DOWN]:
            self.speed -= self.deceleration
        else:
            self.speed *= self.deceleration

        # Update the car's position
        self.rect.x += self.speed

        # Handle collision with the edges of the screen
        if self.rect.left < 0 or self.rect.right > 640:
            self.speed = 0
            self.rect.center = (320, 240)

    def draw(self):
        # Draw the car image
        self.screen.blit(self.image, self.rect)
