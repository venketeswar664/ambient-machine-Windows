import cv2
import numpy as np
tf = __import__('tensorflow')

class FaceDetector:
    """
    In-memory face detector using a TensorFlow SavedModel.

    Usage:
        fd = FaceDetector(model_dir="/path/to/saved_model", score_threshold=0.5)
        detections = fd.detect_faces(frame_np)
        # detections: list of (x1, y1, x2, y2, score)
    """
    def __init__(self, model_dir: str, score_threshold: float = 0.2):
        # Load the TensorFlow SavedModel once
        print(f"[INFO] Loading TensorFlow face model from {model_dir}...")
        self.model = tf.saved_model.load(model_dir)
        self.score_threshold = score_threshold
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def _run_inference(self, image_np: np.ndarray):
        """Run the TF model and return raw boxes and scores."""
        # Convert BGR->RGB
        img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # img_rgb = cv2.resize(img_rgb, (320, 320))
        inp = tf.convert_to_tensor(img_rgb)[tf.newaxis, ...]
        self.model_fn = self.model.signatures['serving_default']
        out = self.model_fn(inp)
        num = int(out['num_detections'][0])
        boxes = out['detection_boxes'][0].numpy()[:num]
        scores = out['detection_scores'][0].numpy()[:num]
        return boxes, scores

    def detect_faces(self, image_np: np.ndarray):
        """
        Detects faces in a single image.

        Args:
            image_np: HxWx3 BGR image as numpy.ndarray.

        Returns:
            List of tuples (x1, y1, x2, y2, score), where coords are pixel values.
        """
        h, w = image_np.shape[:2]
        boxes, scores = self._run_inference(image_np)
        detections = []
        for (ymin, xmin, ymax, xmax), score in zip(boxes, scores):
            if score < self.score_threshold:
                continue
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)
            detections.append((x1, y1, x2, y2, float(score)))
        return detections
