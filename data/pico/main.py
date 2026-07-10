from machine import Pin, PWM
import time
import sys
import json

FW_NAME = "RouletteVision-Production-Failsafe"
FW_VERSION = "2026-07-10-trigger-always-sound-and-servo"

BUZZER_PIN = 0
SERVO_PIN = 2

PERIOD_US = 20000
SERVO_FREQ = 50

CAL_FILE = "calibration.json"

DEFAULT_UP_US = 1000
DEFAULT_TAP_US = 1200
DEFAULT_HOLD_MS = 200

DEFAULT_BUZZ_MS = 450

buzzer = Pin(BUZZER_PIN, Pin.OUT)
buzzer.value(0)

servo = PWM(Pin(SERVO_PIN))
servo.freq(SERVO_FREQ)

servo_up_us = DEFAULT_UP_US
servo_tap_us = DEFAULT_TAP_US
servo_hold_ms = DEFAULT_HOLD_MS


def clamp(value, low, high):
    try:
        value = int(value)
    except Exception:
        value = low
    if value < low:
        return low
    if value > high:
        return high
    return value


def set_servo_us(pulse_us):
    pulse_us = clamp(pulse_us, 500, 2500)
    duty = int((pulse_us / PERIOD_US) * 65535)
    servo.duty_u16(duty)
    print("SERVO GP{} PULSE_US={} DUTY={}".format(SERVO_PIN, pulse_us, duty))


def buzzer_off():
    buzzer.value(0)


def load_calibration():
    global servo_up_us, servo_tap_us, servo_hold_ms

    try:
        with open(CAL_FILE, "r") as f:
            data = json.loads(f.read())

        servo_up_us = clamp(data.get("up_us", DEFAULT_UP_US), 800, 1400)
        servo_tap_us = clamp(data.get("tap_us", DEFAULT_TAP_US), 900, 1600)
        servo_hold_ms = clamp(data.get("hold_ms", DEFAULT_HOLD_MS), 50, 700)

        print("CAL LOADED up={} tap={} hold={}".format(
            servo_up_us, servo_tap_us, servo_hold_ms
        ))
    except Exception as exc:
        print("CAL DEFAULT up={} tap={} hold={} reason={}".format(
            servo_up_us, servo_tap_us, servo_hold_ms, exc
        ))


def save_calibration():
    data = {
        "up_us": servo_up_us,
        "tap_us": servo_tap_us,
        "hold_ms": servo_hold_ms,
    }
    with open(CAL_FILE, "w") as f:
        f.write(json.dumps(data))

    print("CAL SAVED up={} tap={} hold={}".format(
        servo_up_us, servo_tap_us, servo_hold_ms
    ))


def getcal():
    print("CAL up={} tap={} hold={}".format(
        servo_up_us, servo_tap_us, servo_hold_ms
    ))


def setcal(parts):
    global servo_up_us, servo_tap_us, servo_hold_ms

    if len(parts) < 4:
        print("ERROR SETCAL NEEDS: SETCAL up_us tap_us hold_ms")
        return

    servo_up_us = clamp(parts[1], 800, 1400)
    servo_tap_us = clamp(parts[2], 900, 1600)
    servo_hold_ms = clamp(parts[3], 50, 700)

    save_calibration()
    set_servo_us(servo_up_us)
    print("SETCAL DONE up={} tap={} hold={}".format(
        servo_up_us, servo_tap_us, servo_hold_ms
    ))


def beep(ms=DEFAULT_BUZZ_MS):
    ms = clamp(ms, 80, 1500)
    print("BUZZ START GP{} MS={}".format(BUZZER_PIN, ms))
    buzzer.value(1)
    time.sleep_ms(ms)
    buzzer.value(0)
    print("BUZZ DONE")


def tap(pulse_us=None, hold_ms=None):
    if pulse_us is None:
        pulse_us = servo_tap_us
    if hold_ms is None:
        hold_ms = servo_hold_ms

    pulse_us = clamp(pulse_us, 900, 1600)
    hold_ms = clamp(hold_ms, 50, 700)

    print("TAP START GP{} tap={} hold={}".format(SERVO_PIN, pulse_us, hold_ms))
    set_servo_us(pulse_us)
    time.sleep_ms(hold_ms)
    set_servo_us(servo_up_us)
    print("TAP DONE")


def fire():
    """
    Production failsafe action:
    Always buzzer + servo.
    UI flags are ignored here on purpose.
    """

    print("FIRE START up={} tap={} hold={}".format(
        servo_up_us, servo_tap_us, servo_hold_ms
    ))

    # صدا از همان لحظه شروع اکشن روشن می‌شود
    buzzer.value(1)
    print("BUZZ ON")

    # حرکت بازو
    set_servo_us(servo_tap_us)
    time.sleep_ms(servo_hold_ms)
    set_servo_us(servo_up_us)

    # کمی صدا بعد از برگشت بازو هم ادامه پیدا کند
    time.sleep_ms(180)
    buzzer.value(0)
    print("BUZZ OFF")

    # یک beep کوتاه دوم برای اطمینان شنیده‌شدن صدا
    time.sleep_ms(70)
    buzzer.value(1)
    time.sleep_ms(160)
    buzzer.value(0)

    print("FIRE DONE")


def handle(raw):
    raw = raw.strip()
    if not raw:
        return

    print("CMD", raw)
    parts = raw.split()
    cmd = parts[0].upper()

    if cmd == "PING":
        print("PONG")
        return

    if cmd == "STATUS":
        print("STATUS FW={} VERSION={} BUZZER_PIN={} SERVO_PIN={} UP={} TAP={} HOLD={}".format(
            FW_NAME, FW_VERSION, BUZZER_PIN, SERVO_PIN,
            servo_up_us, servo_tap_us, servo_hold_ms
        ))
        return

    if cmd == "GETCAL":
        getcal()
        return

    if cmd == "SETCAL":
        setcal(parts)
        return

    if cmd == "BUZZ":
        ms = int(parts[1]) if len(parts) > 1 else DEFAULT_BUZZ_MS
        beep(ms)
        return

    if cmd == "SERVO":
        pulse = int(parts[1]) if len(parts) > 1 else servo_up_us
        set_servo_us(pulse)
        return

    if cmd == "TAP":
        pulse = int(parts[1]) if len(parts) > 1 else servo_tap_us
        hold = int(parts[2]) if len(parts) > 2 else servo_hold_ms
        tap(pulse, hold)
        return

    if cmd == "FIRE":
        fire()
        return

    if cmd == "TRIGGER":
        # مهم: در حالت production هر TRIGGER حتماً هر دو اکشن را اجرا می‌کند
        fire()
        return

    if cmd == "OFF":
        buzzer_off()
        set_servo_us(servo_up_us)
        print("OFF DONE")
        return

    print("ERROR UNKNOWN_COMMAND", raw)


print("")
print("BOOT {}".format(FW_NAME))
print("VERSION {}".format(FW_VERSION))
print("READY BUZZER_PIN={} SERVO_PIN={}".format(BUZZER_PIN, SERVO_PIN))

load_calibration()
buzzer_off()
set_servo_us(servo_up_us)

while True:
    try:
        line = sys.stdin.readline()
        if line:
            handle(line)
        else:
            time.sleep_ms(20)
    except Exception as exc:
        buzzer_off()
        print("COMMAND ERROR", exc)
