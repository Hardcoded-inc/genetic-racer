from PIL import Image, ImageDraw

# Set the size of the car image
WIDTH = 50
HEIGHT = 30

# Create a blank image
image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))

# Get a drawing context
draw = ImageDraw.Draw(image)

# Draw the car body
body_color = (0, 0, 0)
draw.rectangle((0, 0, WIDTH, HEIGHT), fill=body_color)

# Draw the car wheels
wheel_color = (0, 0, 0)
draw.ellipse((10, 20, 20, 30), fill=wheel_color)
draw.ellipse((30, 20, 40, 30), fill=wheel_color)

# Save the car image
image.save("car.png")
