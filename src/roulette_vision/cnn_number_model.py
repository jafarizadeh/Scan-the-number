from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(x):
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / max(float(e.sum()), 1e-9)


class CNNNumberModel:
    def __init__(self, model_path, img_size=160):
        self.model_path = Path(model_path)
        self.img_size = int(img_size)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, roi_bgr):
        img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)

        return x.astype(np.float32)

    def predict(self, roi_bgr):
        x = self.preprocess(roi_bgr)
        logits = self.session.run([self.output_name], {self.input_name: x})[0][0]
        probs = softmax(logits)

        ranked = np.argsort(probs)[::-1]
        top = int(ranked[0])
        second = int(ranked[1])

        return {
            "label": top,
            "score": float(probs[top]),
            "margin": float(probs[top] - probs[second]),
            "second_label": second,
            "second_score": float(probs[second]),
            "top5": [
                {"label": int(i), "score": float(probs[i])}
                for i in ranked[:5]
            ],
            "source": "cnn_mobilenet_onnx"
        }
