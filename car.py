import pygame
import math

class Car:
    def __init__(self, screen):
        self.screen = screen

        # Load the car image and get its rect
        self.image = pygame.image.load("img/car.png")
        self.rect = self.image.get_rect()

        # Set the car's starting position and angle
        self.rect.center = (320, 240)
        self.angle = 0

        # Set the car's speed
        self.speed = 0

        # Set the car's acceleration, deceleration, and steering
        self.acceleration = 0.1
        self.deceleration = 0.1
        self.steering = 0.1

    def update(self):
        # Handle keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.speed += self.acceleration
        elif keys[pygame.K_DOWN]:
            self.speed -= self.deceleration
        else:
            self.speed *= self.deceleration

        if keys[pygame.K_LEFT]:
            self.angle -= self.steering
        elif keys[pygame.K_RIGHT]:
            self.angle += self.steering

        # Update the car's position and angle
        self.rect.x += self.speed * math.cos(self.angle)
        self.rect.y += self.speed * math.sin(self.angle)

        # Handle collision with the edges of the screen
        if self.rect.left < 0 or self.rect.right > 640:
            self.speed = 0
            self.rect.center = (320, 240)

    def draw(self):
        # Rotate the car image
        rotated_image = pygame.transform.rotate(self.image, self.angle)

        # Get the bounding rect of the rotated image
        rotated_rect = rotated_image.get_rect()

        # Set the rect's center to the original position of the car
        rotated_rect.center = self.rect.center

        # Draw the rotated image
        self.screen.blit(rotated_image, rotated_rect)
