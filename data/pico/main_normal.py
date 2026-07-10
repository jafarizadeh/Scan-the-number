from machine import Pin, PWM
import time
import sys

ACTUATOR_PIN = 15
BUZZER_PIN = 14
ARM_EXTENSION_PERCENT = 35

MIN_PULSE_US = 1000
MAX_PULSE_US = 2400

actuator = PWM(Pin(ACTUATOR_PIN))
actuator.freq(50)

buzzer = Pin(BUZZER_PIN, Pin.OUT)
buzzer.value(0)

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def set_pulse_us(pulse_us):
    pulse_us = clamp(int(pulse_us), MIN_PULSE_US, MAX_PULSE_US)
    period_us = 20000
    duty = int((pulse_us / period_us) * 65535)
    actuator.duty_u16(duty)

def percent_to_pulse(percent):
    percent = clamp(int(percent), 0, 100)
    return MIN_PULSE_US + int((MAX_PULSE_US - MIN_PULSE_US) * percent / 100)

def actuator_in():
    print("ACTUATOR IN")
    set_pulse_us(MIN_PULSE_US)

def actuator_out(percent=100):
    percent = clamp(int(percent), 0, 100)
    print("ACTUATOR OUT", percent)
    set_pulse_us(percent_to_pulse(percent))

def buzzer_on():
    print("BUZZER ON")
    buzzer.value(1)

def buzzer_off():
    print("BUZZER OFF")
    buzzer.value(0)

def beep(duration=0.18):
    buzzer_on()
    time.sleep(max(0.05, float(duration)))
    buzzer_off()

def trigger_action(
    arm_percent=100,
    sound_delay=0.0,
    arm_delay=0.0,
    sound_enabled=True,
    arm_enabled=True,
):
    print("TRIGGER START")

    arm_percent = ARM_EXTENSION_PERCENT
    sound_delay = max(0.0, float(sound_delay))
    arm_delay = max(0.0, float(arm_delay))

    if arm_enabled:
        actuator_in()
        time.sleep(0.25)

    started = time.ticks_ms()
    sound_done = not sound_enabled
    arm_done = not arm_enabled

    while not (sound_done and arm_done):
        elapsed = time.ticks_diff(time.ticks_ms(), started) / 1000.0

        if not sound_done and elapsed >= sound_delay:
            beep(0.18)
            sound_done = True

        if not arm_done and elapsed >= arm_delay:
            actuator_out(arm_percent)
            time.sleep(1.0)
            actuator_in()
            time.sleep(0.25)
            arm_done = True

        time.sleep(0.01)

    buzzer_off()
    print("TRIGGER DONE")


print("Pico ready. Commands: TRIGGER [sound_delay arm_delay sound_enabled arm_enabled], OUT [percent], IN, BUZZ, OFF, TEST")

actuator_in()
buzzer_off()

while True:
    raw = sys.stdin.readline().strip()
    if not raw:
        continue

    parts = raw.split()
    command = parts[0].upper()

    try:
        if command == "TRIGGER":
            sound_delay = float(parts[1]) if len(parts) > 1 else 0.0
            arm_delay = float(parts[2]) if len(parts) > 2 else 0.0
            sound_enabled = bool(int(parts[3])) if len(parts) > 3 else True
            arm_enabled = bool(int(parts[4])) if len(parts) > 4 else True
            trigger_action(ARM_EXTENSION_PERCENT, sound_delay, arm_delay, sound_enabled, arm_enabled)

        elif command == "OUT":
            percent = int(parts[1]) if len(parts) > 1 else 100
            actuator_out(percent)

        elif command == "IN":
            actuator_in()

        elif command == "BUZZ":
            buzzer_on()

        elif command == "OFF":
            buzzer_off()
            actuator_in()

        elif command == "TEST":
            trigger_action(100, 0.0, 0.0, True, True)

        else:
            print("Unknown command:", raw)

    except Exception as exc:
        print("Command error:", exc)
