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
