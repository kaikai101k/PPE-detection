import time
from typing import Tuple

try:
    from rpi_ws281x import PixelStrip, Color
except ImportError:
    PixelStrip = None
    Color = None


class LEDDisplay:
    """
    Simple RGB LED controller for an addressable 8x32 display or strip.

    Designed to be easy to expand:
    - set solid status colors
    - draw individual pixels
    - add animations later
    - add text/icons later
    """

    DEFAULT_COLORS = {
        "off": (0, 0, 0),
        "safe": (0, 255, 0),       # green
        "unsafe": (255, 0, 0),     # red
        "idle": (0, 0, 40),        # dim blue
        "warning": (255, 120, 0),  # orange
    }

    def __init__(
        self,
        width: int = 32,
        height: int = 8,
        pin: int = 18,
        brightness: int = 80,
        freq_hz: int = 800000,
        dma: int = 10,
        invert: bool = False,
        channel: int = 0,
        pixel_order: str = "GRB",
        simulate: bool = False,
    ):
        self.width = width
        self.height = height
        self.num_pixels = width * height
        self.pin = pin
        self.brightness = brightness
        self.freq_hz = freq_hz
        self.dma = dma
        self.invert = invert
        self.channel = channel
        self.pixel_order = pixel_order
        self.simulate = simulate

        self.strip = None
        self.current_state = None

    def begin(self):
        if self.simulate:
            print("[LED] Simulation mode enabled")
            return

        if PixelStrip is None:
            raise ImportError(
                "rpi_ws281x is not installed. Install it or enable simulate=True."
            )

        self.strip = PixelStrip(
            self.num_pixels,
            self.pin,
            self.freq_hz,
            self.dma,
            self.invert,
            self.brightness,
            self.channel,
        )
        self.strip.begin()
        self.clear()

    def _color(self, rgb: Tuple[int, int, int]):
        r, g, b = rgb
        if Color is None:
            return (r, g, b)
        # Most WS2812 setups on Pi use GRB order in practice.
        if self.pixel_order.upper() == "GRB":
            return Color(g, r, b)
        return Color(r, g, b)

    def xy_to_index(self, x: int, y: int) -> int:
        """
        Serpentine mapping for 8x32 matrix.
        Change this if your panel wiring is different.
        """
        if y % 2 == 0:
            return y * self.width + x
        return y * self.width + (self.width - 1 - x)

    def set_pixel(self, x: int, y: int, rgb: Tuple[int, int, int]):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        idx = self.xy_to_index(x, y)
        if self.simulate:
            return

        self.strip.setPixelColor(idx, self._color(rgb))

    def fill(self, rgb: Tuple[int, int, int]):
        if self.simulate:
            print(f"[LED] fill -> {rgb}")
            return

        color = self._color(rgb)
        for i in range(self.num_pixels):
            self.strip.setPixelColor(i, color)
        self.strip.show()

    def clear(self):
        self.fill((0, 0, 0))
        self.current_state = "off"

    def set_state(self, state: str):
        """
        Set a named state color.
        Avoids re-sending the same color repeatedly.
        """
        if state == self.current_state:
            return

        if state not in self.DEFAULT_COLORS:
            raise ValueError(f"Unknown LED state: {state}")

        self.fill(self.DEFAULT_COLORS[state])
        self.current_state = state

    def flash(self, rgb: Tuple[int, int, int], times: int = 3, delay: float = 0.2):
        for _ in range(times):
            self.fill(rgb)
            time.sleep(delay)
            self.clear()
            time.sleep(delay)

    def test_cycle(self):
        self.fill((255, 0, 0))
        time.sleep(0.5)
        self.fill((0, 255, 0))
        time.sleep(0.5)
        self.fill((0, 0, 255))
        time.sleep(0.5)
        self.clear()

    def cleanup(self):
        self.clear()