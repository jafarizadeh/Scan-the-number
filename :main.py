from machine import Pin
import time

pin = Pin(15, Pin.OUT)

print("GP15 DIGITAL OUTPUT TEST")

while True:
    print("GP15 HIGH")
    pin.value(1)
    time.sleep(0.5)

    print("GP15 LOW")
    pin.value(0)
    time.sleep(0.5)
