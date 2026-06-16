#!/usr/bin/env python3
from time import sleep
from pi5neo import Pi5Neo

# --- CONFIG ---
NUM_LEDS = 30           # adjust to your strip length
SPI_DEVICE = "/dev/spidev0.0"
SPI_SPEED_KHZ = 800     # default is fine

# --- INIT ---
neo = Pi5Neo(SPI_DEVICE, NUM_LEDS, SPI_SPEED_KHZ)
print("Initialized Pi5Neo LED strip")

# --- BLINK LOOP ---
while True:
    # Turn all LEDs red
    neo.fill_strip(255, 0, 0)
    neo.update_strip()
    sleep(0.5)

    # Turn all LEDs off
    neo.fill_strip(0, 0, 0)
    neo.update_strip()
    sleep(0.5)
