import board
import neopixel
import time

NUM_PIXELS = 35

pixels = neopixel.NeoPixel(
    board.D18,       # or whatever pin you're using
    NUM_PIXELS,
    brightness=0.5,
    auto_write=False
)

# Red → Green → Blue → Off
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]

for color in colors:
    pixels.fill(color)
    pixels.show()
    time.sleep(1)
