import random
from PIL import Image, ImageDraw

# Set the size of the track image
WIDTH = 640
HEIGHT = 480

# Create a blank image
image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))

# Get a drawing context
draw = ImageDraw.Draw(image)

# Generate a random track
for i in range(50):
    x1 = random.randint(0, WIDTH)
    y1 = random.randint(0, HEIGHT)
    x2 = random.randint(0, WIDTH)
    y2 = random.randint(0, HEIGHT)
    color = (0, 0, 0)
    draw.line((x1, y1, x2, y2), fill=color, width=5)

# Save the track image
image.save("track.png")
