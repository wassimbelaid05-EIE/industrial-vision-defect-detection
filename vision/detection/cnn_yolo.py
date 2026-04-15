"""
CNN-based Defect Classifier + YOLO-style Detector
Uses feature extraction + ML classification (no heavy deep learning dependencies)
Simulates CNN/YOLO behavior with scikit-learn + OpenCV features

Author: Wassim BELAID
"""

import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from simulation.part_generator import DefectType, DefectInfo


@dataclass
class CNNClassification:
    """Result from CNN-based classification."""
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    feature_importance: Dict[str, float]
    is_defective: bool
    method: str = "CNN+RF Ensemble"


@dataclass
class YOLODetection:
    """Result from YOLO-style object detection."""
    detections: List[Dict]  # [{class, confidence, bbox, defect_type}]
    n_objects: int
    processing_time_ms: float


class FeatureExtractor:
    """
    Extracts CNN-like features from images using OpenCV.
    Simulates deep learning feature extraction with:
    - HOG (Histogram of Oriented Gradients)
    - LBP (Local Binary Patterns)
    - Gabor filter responses
    - GLCM texture features
    - Statistical moments
    """

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract feature vector from image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = cv2.resize(gray, (128, 128))

        features = []

        # 1. HOG features (gradient-based, captures shape/texture)
        hog_features = self._compute_hog(gray)
        features.extend(hog_features)

        # 2. LBP texture features
        lbp_features = self._compute_lbp(gray)
        features.extend(lbp_features)

        # 3. Gabor filter responses (frequency/orientation)
        gabor_features = self._compute_gabor(gray)
        features.extend(gabor_features)

        # 4. Statistical features
        stat_features = self._compute_statistics(gray)
        features.extend(stat_features)

        # 5. Frequency domain features (FFT)
        fft_features = self._compute_fft(gray)
        features.extend(fft_features)

        return np.array(features, dtype=np.float32)

    def _compute_hog(self, gray: np.ndarray) -> List[float]:
        """HOG descriptor — captures edge/gradient patterns."""
        # Simplified HOG
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # 9-bin histogram per cell (4×4 grid = 16 cells)
        features = []
        h, w = gray.shape
        cell_h, cell_w = h // 4, w // 4
        for i in range(4):
            for j in range(4):
                cell_mag = mag[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_ang = ang[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                hist, _ = np.histogram(cell_ang, bins=9, range=(0, 180), weights=cell_mag)
                hist = hist / (np.sum(hist) + 1e-6)
                features.extend(hist.tolist())

        return features  # 144 features

    def _compute_lbp(self, gray: np.ndarray) -> List[float]:
        """Local Binary Pattern — texture descriptor."""
        radius = 2
        n_points = 8
        lbp = np.zeros_like(gray, dtype=np.uint8)

        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dx = int(round(radius * np.cos(angle)))
            dy = int(round(-radius * np.sin(angle)))

            shifted = np.zeros_like(gray)
            if dx >= 0 and dy >= 0:
                shifted[dy:, dx:] = gray[:gray.shape[0]-dy, :gray.shape[1]-dx]
            elif dx >= 0 and dy < 0:
                shifted[:dy, dx:] = gray[-dy:, :gray.shape[1]-dx]
            elif dx < 0 and dy >= 0:
                shifted[dy:, :dx] = gray[:gray.shape[0]-dy, -dx:]
            else:
                shifted[:dy, :dx] = gray[-dy:, -dx:]

            lbp += ((shifted >= gray).astype(np.uint8) << i)

        hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
        return (hist / (hist.sum() + 1e-6)).tolist()  # 64 features

    def _compute_gabor(self, gray: np.ndarray) -> List[float]:
        """Gabor filter bank — captures frequency and orientation."""
        features = []
        for theta in [0, 45, 90, 135]:
            for frequency in [0.1, 0.3]:
                kernel = cv2.getGaborKernel(
                    (21, 21), sigma=4, theta=np.radians(theta),
                    lambd=1/frequency, gamma=0.5
                )
                filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
                features.append(float(filtered.mean()))
                features.append(float(filtered.std()))
        return features  # 16 features

    def _compute_statistics(self, gray: np.ndarray) -> List[float]:
        """Statistical image features."""
        from scipy import stats as sp_stats
        flat = gray.ravel().astype(np.float64)
        return [
            float(flat.mean()),
            float(flat.std()),
            float(np.percentile(flat, 25)),
            float(np.percentile(flat, 75)),
            float(sp_stats.skew(flat)),
            float(sp_stats.kurtosis(flat)),
            float(flat.max() - flat.min()),
            float(np.sqrt(np.mean((flat - flat.mean())**2))),
            float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            float(gray.mean()),
        ]  # 10 features

    def _compute_fft(self, gray: np.ndarray) -> List[float]:
        """FFT power spectrum features — detects periodic defects."""
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        log_mag = np.log1p(magnitude)

        h, w = log_mag.shape
        # Radial statistics
        center = (h // 2, w // 2)
        features = []
        for r_start, r_end in [(0, 10), (10, 30), (30, 60), (60, 100)]:
            mask = np.zeros((h, w), dtype=bool)
            for y in range(h):
                for x in range(w):
                    d = np.sqrt((y - center[0])**2 + (x - center[1])**2)
                    if r_start <= d < r_end:
                        mask[y, x] = True
            ring = log_mag[mask]
            if len(ring) > 0:
                features.extend([ring.mean(), ring.std()])
            else:
                features.extend([0.0, 0.0])
        return features  # 8 features

    @property
    def n_features(self) -> int:
        return 144 + 64 + 16 + 10 + 8  # = 242


class CNNClassifier:
    """
    CNN-like defect classifier.

    Uses deep feature extraction (HOG + LBP + Gabor + FFT)
    with ensemble ML classification (RF + GBM + SVM).

    Simulates a fine-tuned ResNet/EfficientNet without
    requiring heavy DL dependencies (works on CPU).
    """

    DEFECT_CLASSES = [
        "no_defect", "scratch", "crack", "dent", "burr",
        "pit", "stain", "chip", "inclusion"
    ]

    def __init__(self):
        self.extractor = FeatureExtractor()
        self._rf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=15,
                                           random_state=42, n_jobs=-1)),
        ])
        self._gbm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                random_state=42)),
        ])
        self._svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=42)),
        ])
        self._is_trained = False
        self._classes = self.DEFECT_CLASSES
        self._training_samples = 0

    def train(self, images: List[np.ndarray], labels: List[str]) -> Dict:
        """Train all classifiers on labeled images."""
        X = np.array([self.extractor.extract(img) for img in images])
        y = np.array(labels)
        self._training_samples = len(X)

        # Train RF (main model)
        self._rf.fit(X, y)

        # Train GBM on subset
        n = min(len(X), 500)
        idx = np.random.choice(len(X), n, replace=False)
        self._gbm.fit(X[idx], y[idx])

        # SVM on smaller subset
        n_svm = min(len(X), 200)
        idx_svm = np.random.choice(len(X), n_svm, replace=False)
        y_svm = y[idx_svm]
        if len(np.unique(y_svm)) >= 2:
            self._svm.fit(X[idx_svm], y_svm)

        self._is_trained = True
        self._classes = list(np.unique(y))

        # CV accuracy
        cv_scores = cross_val_score(self._rf, X, y, cv=min(5, len(np.unique(y))),
                                     scoring="accuracy")
        return {
            "accuracy": round(float(cv_scores.mean()), 4),
            "std": round(float(cv_scores.std()), 4),
            "n_classes": len(np.unique(y)),
            "n_samples": len(X),
            "n_features": X.shape[1],
        }

    def predict(self, image: np.ndarray) -> CNNClassification:
        """Classify image for defects."""
        features = self.extractor.extract(image).reshape(1, -1)

        if not self._is_trained:
            return self._dummy_predict(image)

        # Ensemble prediction
        rf_proba = self._rf.predict_proba(features)[0]
        rf_classes = self._rf.classes_

        try:
            gbm_proba = self._gbm.predict_proba(features)[0]
            gbm_classes = self._gbm.classes_
            # Align probabilities
            all_classes = list(set(rf_classes) | set(gbm_classes))
            ensemble_proba = np.zeros(len(all_classes))
            for i, cls in enumerate(all_classes):
                if cls in rf_classes:
                    ensemble_proba[i] += 0.6 * rf_proba[list(rf_classes).index(cls)]
                if cls in gbm_classes:
                    ensemble_proba[i] += 0.4 * gbm_proba[list(gbm_classes).index(cls)]
        except Exception:
            all_classes = list(rf_classes)
            ensemble_proba = rf_proba

        pred_idx = np.argmax(ensemble_proba)
        pred_class = all_classes[pred_idx]
        confidence = float(ensemble_proba[pred_idx])

        class_probs = {str(c): round(float(p), 4)
                      for c, p in zip(all_classes, ensemble_proba)}

        # Feature importance
        if hasattr(self._rf.named_steps["clf"], "feature_importances_"):
            importances = self._rf.named_steps["clf"].feature_importances_
            feat_names = ["HOG", "LBP", "Gabor", "Statistics", "FFT"]
            feat_sizes = [144, 64, 16, 10, 8]
            feat_imp = {}
            start = 0
            for name, size in zip(feat_names, feat_sizes):
                feat_imp[name] = round(float(importances[start:start+size].sum()), 4)
                start += size
        else:
            feat_imp = {}

        return CNNClassification(
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            class_probabilities=class_probs,
            feature_importance=feat_imp,
            is_defective=pred_class != "no_defect",
            method="CNN Ensemble (RF+GBM+features)",
        )

    def _dummy_predict(self, image: np.ndarray) -> CNNClassification:
        """Quick prediction without training using heuristics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = gray.mean()

        # Simple heuristic scoring
        if laplacian > 200:
            pred = "scratch"
            conf = 0.72
        elif mean_brightness < 100:
            pred = "crack"
            conf = 0.68
        else:
            pred = "no_defect"
            conf = 0.85

        classes = ["no_defect", "scratch", "crack", "dent", "stain"]
        probs = {c: round(np.random.uniform(0, 0.2), 3) for c in classes}
        probs[pred] = round(conf, 3)

        return CNNClassification(
            predicted_class=pred, confidence=conf,
            class_probabilities=probs, feature_importance={},
            is_defective=pred != "no_defect",
            method="Heuristic (pre-training)",
        )

    @property
    def is_trained(self) -> bool:
        return self._is_trained


class YOLOStyleDetector:
    """
    YOLO-style defect localization.

    Divides image into grid cells (similar to YOLOv3/v5),
    predicts bounding boxes and class labels per cell.

    Uses sliding window with classical CV + ML for each region.
    Simulates YOLOv8 behavior without PyTorch dependency.
    """

    GRID_SIZE = 8  # 8×8 grid (similar to YOLO grid)
    CONFIDENCE_THRESHOLD = 0.45
    NMS_THRESHOLD = 0.40

    def __init__(self, classifier: CNNClassifier):
        self.classifier = classifier
        self._cell_size = 512 // self.GRID_SIZE  # 64px per cell

    def detect(self, image: np.ndarray) -> YOLODetection:
        """
        YOLO-style detection on image.
        Returns bounding boxes with class predictions.
        """
        import time
        t0 = time.time()

        h, w = image.shape[:2]
        detections = []
        cell_h = h // self.GRID_SIZE
        cell_w = w // self.GRID_SIZE

        # Process each grid cell
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                y1, x1 = i * cell_h, j * cell_w
                y2, x2 = y1 + cell_h, x1 + cell_w

                cell = image[y1:y2, x1:x2]

                # Quick anomaly check
                gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                local_std = gray_cell.std()
                local_mean = gray_cell.mean()

                # Only classify cells on the part (not background)
                if local_mean < 40:
                    continue

                # Quick CNN prediction on cell
                if local_std > 15 or local_mean > 50:
                    # Upscale cell for better feature extraction
                    cell_resized = cv2.resize(cell, (128, 128))
                    result = self.classifier.predict(cell_resized)

                    if result.is_defective and result.confidence > self.CONFIDENCE_THRESHOLD:
                        detections.append({
                            "class": result.predicted_class,
                            "confidence": round(result.confidence, 3),
                            "bbox": (x1, y1, cell_w, cell_h),  # (x, y, w, h)
                            "center": (x1 + cell_w//2, y1 + cell_h//2),
                            "grid_cell": (i, j),
                            "defect_type": result.predicted_class,
                        })

        # Non-Maximum Suppression
        detections = self._nms(detections)

        t1 = time.time()
        return YOLODetection(
            detections=detections,
            n_objects=len(detections),
            processing_time_ms=round((t1 - t0) * 1000, 1),
        )

    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """Simple Non-Maximum Suppression."""
        if not detections:
            return []

        # Sort by confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        kept = []
        for det in detections:
            x1, y1, w, h = det["bbox"]
            overlap = False
            for kept_det in kept:
                kx1, ky1, kw, kh = kept_det["bbox"]
                # Compute IoU
                ix1 = max(x1, kx1)
                iy1 = max(y1, ky1)
                ix2 = min(x1 + w, kx1 + kw)
                iy2 = min(y1 + h, ky1 + kh)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = w * h + kw * kh - inter
                    iou = inter / max(union, 1)
                    if iou > self.NMS_THRESHOLD:
                        overlap = True
                        break
            if not overlap:
                kept.append(det)

        return kept

    def annotate(self, image: np.ndarray, result: YOLODetection) -> np.ndarray:
        """Draw YOLO-style bounding boxes on image."""
        annotated = image.copy()
        colors = {
            "scratch": (0, 0, 255), "crack": (0, 0, 180),
            "dent": (0, 165, 255), "burr": (0, 255, 255),
            "pit": (255, 0, 0), "stain": (255, 165, 0),
            "chip": (180, 0, 255), "inclusion": (255, 0, 180),
        }

        for det in result.detections:
            x, y, w, h = det["bbox"]
            color = colors.get(det["class"], (255, 255, 0))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(annotated, label, (x + 2, y - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # YOLO grid overlay
        h_img, w_img = annotated.shape[:2]
        grid = self.GRID_SIZE
        for i in range(1, grid):
            cv2.line(annotated, (i * w_img // grid, 0), (i * w_img // grid, h_img),
                    (50, 50, 50), 1)
            cv2.line(annotated, (0, i * h_img // grid), (w_img, i * h_img // grid),
                    (50, 50, 50), 1)

        return annotated
