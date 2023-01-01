import pygame
import math
from utils import scale_image


START_POSITION = (180, 200)
START_DIRECTION = 0
DRIFT_VEL_THRESHOLD = 0.5
DRIFT_FRICTION = 0.87


class Car:
    def __init__(self, screen):
        self.screen = screen

        # Set the car's starting position and direction
        image = pygame.image.load("img/car.png")
        image = pygame.transform.scale(image, (48, 96))
        self.image = scale_image(image, 0.4)
        self.rect = self.image.get_rect()
        self.rect.center = START_POSITION
        self.direction = START_DIRECTION
        self.drift_momentum = 0

        # Set the car's velocity
        self.vel = 0

        # Set the car's acceleration, deceleration, and steering
        self.max_vel = 20
        self.acceleration_rate = 0.2
        self.deceleration_rate = 0.1
        self.braking_rate = 0.7
        self.steering = 7


    def accelerate(self):
        self.vel = min(self.vel + self.acceleration_rate, self.max_vel)

    def brake(self):
        self.vel = max(self.vel - self.braking_rate, 0)

    def decelerate(self):
        self.vel = max(self.vel - self.deceleration_rate, 0)

    def move(self):

        y_displacement = 0
        x_displacement = 0

        dir_radians = math.radians(self.direction)
        x_displacement += self.vel * math.sin(dir_radians)
        y_displacement += self.vel * math.cos(dir_radians)

        drift_direction = self.direction + 90
        drift_dir_radians = math.radians(drift_direction)
        x_displacement += self.drift_momentum * math.sin(drift_dir_radians)
        x_displacement += self.drift_momentum * math.cos(drift_dir_radians)
        self.drift_momentum *= DRIFT_FRICTION

        self.rect.x -= x_displacement
        self.rect.y -= y_displacement


    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.image)
        offset = (int(self.rect.x - x), int(self.rect.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.vel = 0
        self.rect.center = START_POSITION
        self.prev_angle = START_DIRECTION
        self.direction = START_DIRECTION

    def get_drift_amount(self):
        if self.vel > DRIFT_VEL_THRESHOLD:
            return self.vel * self.steering / 70.0
        else:
            return 0

    def turn_left(self):
        self.direction += self.steering
        self.drift_momentum -= self.get_drift_amount()

    def turn_right(self):
        self.direction -= self.steering
        self.drift_momentum += self.get_drift_amount()


    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.accelerate()
        elif keys[pygame.K_DOWN]:
            self.brake()
        else:
            self.decelerate()


        if keys[pygame.K_LEFT]:
            self.turn_left()
        elif keys[pygame.K_RIGHT]:
            self.turn_right()


        self.move()

        # Handle collision with the edges of the screen
        screen_width, screen_height = pygame.display.get_surface().get_size()

        if (self.rect.left < 0
        or self.rect.right > screen_width
        or self.rect.top < 0
        or self.rect.bottom > screen_height):
            self.vel = 0
            self.rect.center = START_POSITION
            self.direction = START_DIRECTION


    def draw(self):
        # Rotate the car image
        rotated_image = pygame.transform.rotate(self.image, self.direction)

        # Get the bounding rect of the rotated image
        rotated_rect = rotated_image.get_rect()

        # Set the rect's center to the original position of the car
        rotated_rect.center = self.rect.center

        # Draw the rotated image
        self.screen.blit(rotated_image, rotated_rect)
