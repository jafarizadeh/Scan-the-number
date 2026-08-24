from machine import Pin, PWM
import time
import sys
import json

FW_NAME = "RouletteVision-Actuonix-Production"
FW_VERSION = "2026-07-12-actuonix-gp17-synchronized"

BUZZER_PIN = 0
ACTUATOR_PIN = 17

PWM_FREQUENCY = 50
PWM_PERIOD_US = 20000

CAL_FILE = "calibration.json"

# Safe initial positions for Actuonix L12-R.
DEFAULT_RETRACT_US = 1300
DEFAULT_PRESS_US = 1700

# Time allowed for the actuator to remain at the press position.
DEFAULT_PRESS_HOLD_MS = 900

# Time allowed for returning to the retracted position.
DEFAULT_RETURN_MS = 900

DEFAULT_BUZZ_MS = 900

MIN_POSITION_US = 1000
MAX_POSITION_US = 2000

buzzer = Pin(BUZZER_PIN, Pin.OUT)
buzzer.value(0)

actuator_pwm = None

retract_us = DEFAULT_RETRACT_US
press_us = DEFAULT_PRESS_US
press_hold_ms = DEFAULT_PRESS_HOLD_MS
return_ms = DEFAULT_RETURN_MS


def clamp(value, minimum, maximum):
    try:
        value = int(value)
    except Exception:
        value = minimum

    if value < minimum:
        return minimum

    if value > maximum:
        return maximum

    return value


def enable_actuator():
    global actuator_pwm

    if actuator_pwm is None:
        actuator_pwm = PWM(Pin(ACTUATOR_PIN))
        actuator_pwm.freq(PWM_FREQUENCY)


def disable_actuator():
    global actuator_pwm

    if actuator_pwm is not None:
        try:
            actuator_pwm.deinit()
        except Exception:
            pass

        actuator_pwm = None

    # Leave the signal pin in a known low state.
    Pin(ACTUATOR_PIN, Pin.OUT).value(0)


def set_actuator_us(pulse_us):
    pulse_us = clamp(pulse_us, MIN_POSITION_US, MAX_POSITION_US)

    enable_actuator()

    duty = int((pulse_us / PWM_PERIOD_US) * 65535)
    actuator_pwm.duty_u16(duty)

    print(
        "ACTUATOR GP{} PULSE_US={} DUTY={}".format(
            ACTUATOR_PIN,
            pulse_us,
            duty,
        )
    )


def buzzer_off():
    buzzer.value(0)


def load_calibration():
    global retract_us
    global press_us
    global press_hold_ms
    global return_ms

    try:
        with open(CAL_FILE, "r") as file:
            data = json.loads(file.read())

        # Support both the new Actuonix names and the older servo names.
        retract_us = clamp(
            data.get("retract_us", data.get("up_us", DEFAULT_RETRACT_US)),
            MIN_POSITION_US,
            MAX_POSITION_US,
        )

        press_us = clamp(
            data.get("press_us", data.get("tap_us", DEFAULT_PRESS_US)),
            MIN_POSITION_US,
            MAX_POSITION_US,
        )

        press_hold_ms = clamp(
            data.get(
                "press_hold_ms",
                data.get("hold_ms", DEFAULT_PRESS_HOLD_MS),
            ),
            100,
            3000,
        )

        return_ms = clamp(
            data.get("return_ms", DEFAULT_RETURN_MS),
            100,
            3000,
        )

        print(
            "CAL LOADED retract={} press={} hold={} return={}".format(
                retract_us,
                press_us,
                press_hold_ms,
                return_ms,
            )
        )

    except Exception as exc:
        print(
            "CAL DEFAULT retract={} press={} hold={} return={} reason={}".format(
                retract_us,
                press_us,
                press_hold_ms,
                return_ms,
                exc,
            )
        )


def save_calibration():
    data = {
        "retract_us": retract_us,
        "press_us": press_us,
        "press_hold_ms": press_hold_ms,
        "return_ms": return_ms,

        # Compatibility with the current calibration interface.
        "up_us": retract_us,
        "tap_us": press_us,
        "hold_ms": press_hold_ms,
    }

    with open(CAL_FILE, "w") as file:
        file.write(json.dumps(data))

    print(
        "CAL SAVED retract={} press={} hold={} return={}".format(
            retract_us,
            press_us,
            press_hold_ms,
            return_ms,
        )
    )


def print_calibration():
    print(
        "CAL retract={} press={} hold={} return={}".format(
            retract_us,
            press_us,
            press_hold_ms,
            return_ms,
        )
    )


def set_calibration(parts):
    global retract_us
    global press_us
    global press_hold_ms
    global return_ms

    if len(parts) < 4:
        print(
            "ERROR SETCAL NEEDS: "
            "SETCAL retract_us press_us hold_ms [return_ms]"
        )
        return

    retract_us = clamp(
        parts[1],
        MIN_POSITION_US,
        MAX_POSITION_US,
    )

    press_us = clamp(
        parts[2],
        MIN_POSITION_US,
        MAX_POSITION_US,
    )

    press_hold_ms = clamp(
        parts[3],
        100,
        3000,
    )

    if len(parts) >= 5:
        return_ms = clamp(
            parts[4],
            100,
            3000,
        )

    save_calibration()

    set_actuator_us(retract_us)
    time.sleep_ms(return_ms)
    disable_actuator()

    print(
        "SETCAL DONE retract={} press={} hold={} return={}".format(
            retract_us,
            press_us,
            press_hold_ms,
            return_ms,
        )
    )


def beep(milliseconds=DEFAULT_BUZZ_MS):
    milliseconds = clamp(milliseconds, 80, 3000)

    print(
        "BUZZ START GP{} MS={}".format(
            BUZZER_PIN,
            milliseconds,
        )
    )

    buzzer.value(1)
    time.sleep_ms(milliseconds)
    buzzer.value(0)

    print("BUZZ DONE")


def move_to_position(pulse_us, settle_ms=700):
    pulse_us = clamp(
        pulse_us,
        MIN_POSITION_US,
        MAX_POSITION_US,
    )

    settle_ms = clamp(
        settle_ms,
        100,
        3000,
    )

    set_actuator_us(pulse_us)
    time.sleep_ms(settle_ms)


def tap(
    target_us=None,
    hold_ms=None,
):
    if target_us is None:
        target_us = press_us

    if hold_ms is None:
        hold_ms = press_hold_ms

    target_us = clamp(
        target_us,
        MIN_POSITION_US,
        MAX_POSITION_US,
    )

    hold_ms = clamp(
        hold_ms,
        100,
        3000,
    )

    print(
        "TAP START GP{} retract={} press={} hold={} return={}".format(
            ACTUATOR_PIN,
            retract_us,
            target_us,
            hold_ms,
            return_ms,
        )
    )

    # Extend toward the screen/button.
    set_actuator_us(target_us)
    time.sleep_ms(hold_ms)

    # Return to the safe position.
    set_actuator_us(retract_us)
    time.sleep_ms(return_ms)

    # Stop sending PWM so the actuator is not continuously driven.
    disable_actuator()

    print("TAP DONE")


def fire():
    print(
        "FIRE START retract={} press={} hold={} return={}".format(
            retract_us,
            press_us,
            press_hold_ms,
            return_ms,
        )
    )

    # The buzzer starts immediately before the actuator command.
    # Therefore sound and actuator activation start together.
    buzzer.value(1)
    print("BUZZ ON")

    # Extend actuator.
    set_actuator_us(press_us)
    print("ACTUATOR PRESS")
    time.sleep_ms(press_hold_ms)

    # Retract actuator while the buzzer remains active.
    set_actuator_us(retract_us)
    print("ACTUATOR RETRACT")
    time.sleep_ms(return_ms)

    # Stop both outputs safely.
    buzzer.value(0)
    print("BUZZ OFF")

    disable_actuator()
    print("ACTUATOR PWM RELEASED")

    print("FIRE DONE")



def fire_selective(sound_enabled=True, actuator_enabled=True):
    sound_enabled = bool(sound_enabled)
    actuator_enabled = bool(actuator_enabled)

    print(
        "TRIGGER START sound={} tapper={}".format(
            int(sound_enabled),
            int(actuator_enabled),
        )
    )

    if not sound_enabled and not actuator_enabled:
        buzzer.value(0)
        disable_actuator()
        print("TRIGGER DONE")
        return

    if sound_enabled:
        buzzer.value(1)
        print("BUZZ ON")

    if actuator_enabled:
        set_actuator_us(press_us)
        print("ACTUATOR PRESS")
        time.sleep_ms(press_hold_ms)

        set_actuator_us(retract_us)
        print("ACTUATOR RETRACT")
        time.sleep_ms(return_ms)

    elif sound_enabled:
        time.sleep_ms(DEFAULT_BUZZ_MS)

    if sound_enabled:
        buzzer.value(0)
        print("BUZZ OFF")

    if actuator_enabled:
        disable_actuator()
        print("ACTUATOR PWM RELEASED")

    print("TRIGGER DONE")

def safe_off():
    buzzer_off()

    # Return to the safe/retracted position before releasing PWM.
    set_actuator_us(retract_us)
    time.sleep_ms(return_ms)
    disable_actuator()

    print("OFF DONE")


def handle_command(raw):
    raw = raw.strip()

    if not raw:
        return

    print("CMD", raw)

    parts = raw.split()
    command = parts[0].upper()

    if command == "PING":
        print("PONG")
        return

    if command == "STATUS":
        print(
            "STATUS FW={} VERSION={} BUZZER_PIN={} ACTUATOR_PIN={} "
            "RETRACT={} PRESS={} HOLD={} RETURN={}".format(
                FW_NAME,
                FW_VERSION,
                BUZZER_PIN,
                ACTUATOR_PIN,
                retract_us,
                press_us,
                press_hold_ms,
                return_ms,
            )
        )
        return

    if command == "GETCAL":
        print_calibration()
        return

    if command == "SETCAL":
        set_calibration(parts)
        return

    if command == "BUZZ":
        duration = (
            int(parts[1])
            if len(parts) > 1
            else DEFAULT_BUZZ_MS
        )

        beep(duration)
        return

    if command in ("SERVO", "ACTUATOR"):
        position = (
            int(parts[1])
            if len(parts) > 1
            else retract_us
        )

        move_to_position(position)
        return

    if command == "TAP":
        position = (
            int(parts[1])
            if len(parts) > 1
            else press_us
        )

        duration = (
            int(parts[2])
            if len(parts) > 2
            else press_hold_ms
        )

        tap(position, duration)
        return

    if command == "FIRE":
        fire()
        return

    if command == "TRIGGER":
        try:
            sound_enabled = bool(int(parts[3])) if len(parts) > 3 else True
            tapper_enabled = bool(int(parts[4])) if len(parts) > 4 else True
        except Exception:
            print("ERROR TRIGGER NEEDS: TRIGGER sound_delay tapper_delay sound_enabled tapper_enabled")
            return

        fire_selective(
            sound_enabled=sound_enabled,
            actuator_enabled=tapper_enabled,
        )
        return

    if command == "OFF":
        safe_off()
        return

    print("ERROR UNKNOWN_COMMAND", raw)


print("")
print("BOOT {}".format(FW_NAME))
print("VERSION {}".format(FW_VERSION))
print(
    "READY BUZZER_PIN={} ACTUATOR_PIN={}".format(
        BUZZER_PIN,
        ACTUATOR_PIN,
    )
)

load_calibration()
buzzer_off()

# Move once to the safe/retracted position at startup.
set_actuator_us(retract_us)
time.sleep_ms(return_ms)
disable_actuator()

while True:
    try:
        line = sys.stdin.readline()

        if line:
            handle_command(line)
        else:
            time.sleep_ms(20)

    except Exception as exc:
        buzzer_off()
        disable_actuator()
        print("COMMAND ERROR", exc)
