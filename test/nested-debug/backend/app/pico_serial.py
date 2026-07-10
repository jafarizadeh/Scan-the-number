import time
import glob
import serial
from serial import SerialException

class PicoSerial:
    def __init__(self, port: str = "auto", baud: int = 115200):
        self.port_setting = port
        self.baud = baud
        self.ser = None
        self.last_error = None

    def find_port(self) -> str | None:
        if self.port_setting and self.port_setting != "auto":
            return self.port_setting

        candidates = []
        candidates.extend(glob.glob("/dev/ttyACM*"))
        candidates.extend(glob.glob("/dev/ttyUSB*"))
        candidates.extend(glob.glob("COM*"))

        return candidates[0] if candidates else None

    def connect(self) -> bool:
        if self.ser and self.ser.is_open:
            return True

        port = self.find_port()
        if not port:
            self.last_error = "Pico serial port not found. Try setting PICO_PORT=/dev/ttyACM0"
            return False

        try:
            self.ser = serial.Serial(port, self.baud, timeout=1, write_timeout=1)
            time.sleep(2.0)  # Pico may reset after serial open
            self.last_error = None
            return True
        except SerialException as exc:
            self.last_error = f"Could not open Pico serial port {port}: {exc}"
            self.ser = None
            return False

    def send(self, command: str) -> bool:
        if not self.connect():
            return False

        try:
            payload = (command.strip() + "\n").encode("utf-8")
            self.ser.write(payload)
            self.ser.flush()
            return True
        except SerialException as exc:
            self.last_error = f"Could not send command to Pico: {exc}"
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            return False

    def status(self) -> dict:
        port = None
        open_state = False
        if self.ser:
            port = self.ser.port
            open_state = self.ser.is_open
        return {
            "port_setting": self.port_setting,
            "connected_port": port,
            "is_open": open_state,
            "last_error": self.last_error,
        }
