import time
import cv2


class FrameSource:
    def __init__(self, camera_cfg, source=None):
        self.camera_cfg = camera_cfg
        self.source = source
        self.picam2 = None
        self.cap = None

    def _build_picamera_controls(self):
        user_controls = self.camera_cfg.get("controls", {}) or {}
        control_values = {}

        try:
            from libcamera import controls
        except Exception:
            controls = None

        af_mode = str(user_controls.get("af_mode", "")).lower().strip()

        if controls is not None:
            if af_mode == "continuous":
                control_values["AfMode"] = controls.AfModeEnum.Continuous
            elif af_mode == "auto":
                control_values["AfMode"] = controls.AfModeEnum.Auto
            elif af_mode == "manual":
                control_values["AfMode"] = controls.AfModeEnum.Manual
                if "lens_position" in user_controls:
                    control_values["LensPosition"] = float(user_controls["lens_position"])

        if "exposure_time" in user_controls and "analogue_gain" in user_controls:
            control_values["AeEnable"] = False
            control_values["ExposureTime"] = int(user_controls["exposure_time"])
            control_values["AnalogueGain"] = float(user_controls["analogue_gain"])

        if "awb_enable" in user_controls:
            control_values["AwbEnable"] = bool(user_controls["awb_enable"])

        if "colour_gains" in user_controls:
            gains = user_controls["colour_gains"]
            if isinstance(gains, list) and len(gains) == 2:
                control_values["AwbEnable"] = False
                control_values["ColourGains"] = (float(gains[0]), float(gains[1]))

        return control_values

    def start(self):
        if self.source is not None:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {self.source}")
            return

        from picamera2 import Picamera2

        width = int(self.camera_cfg["width"])
        height = int(self.camera_cfg["height"])
        fps = int(self.camera_cfg["fps"])

        self.picam2 = Picamera2()

        base_controls = {
            "FrameRate": fps
        }

        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls=base_controls
        )

        self.picam2.configure(config)

        extra_controls = self._build_picamera_controls()
        if extra_controls:
            try:
                self.picam2.set_controls(extra_controls)
                print("Camera controls applied:", extra_controls)
            except Exception as e:
                print("WARNING: could not apply camera controls:", e)

        self.picam2.start()
        time.sleep(1.5)

    def set_controls(self, controls_dict):
        if self.picam2 is None:
            return
        self.picam2.set_controls(controls_dict)

    def read(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                return None
            return frame

        if self.picam2 is None:
            raise RuntimeError("Camera is not started")

        frame_rgb = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def metadata(self):
        if self.picam2 is None:
            return {}
        try:
            return self.picam2.capture_metadata()
        except Exception:
            return {}

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.picam2 is not None:
            self.picam2.stop()
