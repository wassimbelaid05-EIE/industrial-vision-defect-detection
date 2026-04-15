"""
Classical Computer Vision Defect Detector
OpenCV-based detection: morphology, blob detection, edge analysis

Standards: ISO 10110 (optics), ISO 1302 (surface texture)
Author: Wassim BELAID
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from simulation.part_generator import DefectType, DefectInfo


@dataclass
class DetectionResult:
    """Result from classical CV detection."""
    detected_defects: List[DefectInfo]
    anomaly_score: float         # 0-100 (higher = more defects)
    surface_roughness_ra: float  # Simulated Ra (µm)
    edge_quality_score: float    # 0-100
    contrast_score: float        # 0-100
    processing_time_ms: float
    method: str = "classical_cv"
    image_quality: str = "good"  # "good", "blurry", "overexposed"
    annotations: np.ndarray = field(default_factory=lambda: np.zeros((512,512,3)))


class ClassicalDefectDetector:
    """
    Classical Computer Vision defect detector.

    Pipeline:
    1. Preprocessing (bilateral filter, CLAHE)
    2. Background subtraction & part segmentation
    3. Surface analysis (texture, roughness estimation)
    4. Edge quality analysis (Canny, contour analysis)
    5. Blob detection (dark/bright anomalies)
    6. Morphological analysis (scratches, cracks)
    7. Anomaly scoring

    ISO 10110: Optical surface defect notation
    ISO 1302: Surface texture Ra, Rz measurement
    """

    def __init__(self):
        # CLAHE for contrast enhancement
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Blob detector for pits, inclusions
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 8000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByColor = True
        params.blobColor = 0  # Dark blobs
        self._blob_detector_dark = cv2.SimpleBlobDetector_create(params)

        params2 = cv2.SimpleBlobDetector_Params()
        params2.filterByArea = True
        params2.minArea = 20
        params2.maxArea = 5000
        params2.filterByColor = True
        params2.blobColor = 255  # Bright blobs
        self._blob_detector_bright = cv2.SimpleBlobDetector_create(params2)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Full defect detection pipeline.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            DetectionResult with all detected defects and scores
        """
        import time
        t0 = time.time()

        h, w = image.shape[:2]
        annotated = image.copy()
        detected_defects = []

        # 1. Image quality check
        quality = self._assess_image_quality(image)

        # 2. Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # 3. Part segmentation
        part_mask = self._segment_part(denoised)

        if np.sum(part_mask) < 1000:
            return DetectionResult([], 0, 0, 100, 100, 0, "No part detected")

        # 4. Surface analysis
        ra = self._estimate_surface_roughness(denoised, part_mask)

        # 5. Edge quality
        edge_score, edge_defects, annotated = self._analyze_edges(
            denoised, part_mask, annotated)
        detected_defects.extend(edge_defects)

        # 6. Blob detection (pits, inclusions)
        blob_defects, annotated = self._detect_blobs(denoised, part_mask, annotated)
        detected_defects.extend(blob_defects)

        # 7. Scratch/crack detection (linear morphology)
        linear_defects, annotated = self._detect_linear_defects(denoised, part_mask, annotated)
        detected_defects.extend(linear_defects)

        # 8. Texture anomalies (stains, dents)
        texture_defects, annotated = self._detect_texture_anomalies(denoised, part_mask, annotated)
        detected_defects.extend(texture_defects)

        # 9. Anomaly score
        anomaly_score = self._compute_anomaly_score(detected_defects, part_mask, denoised)

        # Contrast score
        roi = gray[part_mask > 0]
        contrast = float(roi.std()) if len(roi) > 0 else 0

        t1 = time.time()

        return DetectionResult(
            detected_defects=detected_defects,
            anomaly_score=round(anomaly_score, 2),
            surface_roughness_ra=round(ra, 3),
            edge_quality_score=round(edge_score, 1),
            contrast_score=round(min(100, contrast * 2), 1),
            processing_time_ms=round((t1 - t0) * 1000, 1),
            method="OpenCV Classical",
            image_quality=quality,
            annotations=annotated,
        )

    def _assess_image_quality(self, image: np.ndarray) -> str:
        """Assess image quality (blur, exposure)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = gray.mean()

        if laplacian_var < 50:
            return "blurry"
        elif mean_brightness > 230:
            return "overexposed"
        elif mean_brightness < 20:
            return "underexposed"
        return "good"

    def _segment_part(self, gray: np.ndarray) -> np.ndarray:
        """Segment part from dark background."""
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _estimate_surface_roughness(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """
        Estimate surface roughness Ra from image texture.
        Based on: Ra ∝ std(high-frequency components)
        """
        if np.sum(mask) == 0:
            return 0.0
        # High-pass filter to extract surface texture
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        high_freq = gray.astype(float) - blur.astype(float)
        roi = high_freq[mask > 0]
        ra = float(np.mean(np.abs(roi))) * 0.015  # Scale to µm
        return max(0.01, min(50, ra))

    def _analyze_edges(self, gray: np.ndarray, mask: np.ndarray,
                        annotated: np.ndarray) -> Tuple[float, List[DefectInfo], np.ndarray]:
        """Detect edge defects (chips, burrs)."""
        edges = cv2.Canny(gray, 30, 100)
        edges_masked = cv2.bitwise_and(edges, edges, mask=mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defects = []
        edge_quality = 100.0

        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                continue

            # Hull defects (burrs, chips)
            hull = cv2.convexHull(cnt, returnPoints=False)
            if hull is not None and len(hull) > 3 and len(cnt) > 3:
                try:
                    hull_defects = cv2.convexityDefects(cnt, hull)
                    if hull_defects is not None:
                        for defect in hull_defects:
                            s, e, f, d = defect[0]
                            depth = d / 256.0
                            if depth > 15:  # Significant defect
                                fpt = tuple(cnt[f][0])
                                d_type = DefectType.CHIP if depth > 30 else DefectType.BURR
                                severity = "high" if depth > 30 else "medium"
                                defects.append(DefectInfo(
                                    defect_type=d_type, severity=severity,
                                    location=fpt, size=int(depth),
                                    confidence=min(0.95, depth / 50),
                                    bbox=(fpt[0] - 20, fpt[1] - 20, 40, 40),
                                ))
                                cv2.circle(annotated, fpt, 15, (0, 0, 255), 2)
                                cv2.putText(annotated, d_type.value, (fpt[0]+5, fpt[1]-5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                                edge_quality -= depth / 5
                except cv2.error:
                    pass

        return max(0, edge_quality), defects, annotated

    def _detect_blobs(self, gray: np.ndarray, mask: np.ndarray,
                       annotated: np.ndarray) -> Tuple[List[DefectInfo], np.ndarray]:
        """Detect pits, inclusions, stains using blob detection."""
        # Apply mask
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
        defects = []

        # Dark blobs (pits, inclusions)
        keypoints_dark = self._blob_detector_dark.detect(gray_masked)
        for kp in keypoints_dark:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            severity = "critical" if size > 30 else ("high" if size > 15 else "medium")
            d_type = DefectType.PIT if size < 25 else DefectType.INCLUSION
            defects.append(DefectInfo(
                defect_type=d_type, severity=severity,
                location=(x, y), size=size,
                confidence=min(0.95, 0.7 + size / 100),
                bbox=(x - size, y - size, size * 2, size * 2),
            ))
            cv2.circle(annotated, (x, y), max(size, 10), (0, 165, 255), 2)
            cv2.putText(annotated, d_type.value, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

        return defects, annotated

    def _detect_linear_defects(self, gray: np.ndarray, mask: np.ndarray,
                                 annotated: np.ndarray) -> Tuple[List[DefectInfo], np.ndarray]:
        """Detect scratches and cracks using morphological operations."""
        defects = []

        # Tophat transform to enhance linear features
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        tophat_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
        tophat_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v)
        linear = cv2.add(tophat_h, tophat_v)

        # Apply mask and threshold
        linear_masked = cv2.bitwise_and(linear, linear, mask=mask)
        _, thresh = cv2.threshold(linear_masked, 20, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w_comp = stats[i, cv2.CC_STAT_WIDTH]
            h_comp = stats[i, cv2.CC_STAT_HEIGHT]
            x_comp = stats[i, cv2.CC_STAT_LEFT]
            y_comp = stats[i, cv2.CC_STAT_TOP]

            if area < 50:
                continue

            aspect = max(w_comp, h_comp) / max(min(w_comp, h_comp), 1)
            cx, cy = int(centroids[i][0]), int(centroids[i][1])

            if aspect > 3:  # Linear feature = scratch/crack
                d_type = DefectType.CRACK if area > 500 else DefectType.SCRATCH
                severity = "critical" if area > 800 else ("high" if area > 300 else "medium")
                defects.append(DefectInfo(
                    defect_type=d_type, severity=severity,
                    location=(cx, cy), size=int(np.sqrt(area)),
                    confidence=min(0.92, 0.65 + area / 2000),
                    bbox=(x_comp, y_comp, w_comp, h_comp),
                ))
                cv2.rectangle(annotated, (x_comp, y_comp),
                             (x_comp + w_comp, y_comp + h_comp), (255, 50, 50), 2)
                cv2.putText(annotated, d_type.value, (x_comp, y_comp - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 50, 50), 1)

        return defects, annotated

    def _detect_texture_anomalies(self, gray: np.ndarray, mask: np.ndarray,
                                    annotated: np.ndarray) -> Tuple[List[DefectInfo], np.ndarray]:
        """Detect stains and dents from texture anomalies."""
        defects = []

        # Local texture variance map
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        diff = cv2.absdiff(gray, blur)
        diff_masked = cv2.bitwise_and(diff, diff, mask=mask)

        # Regions with unusual texture
        _, thresh = cv2.threshold(diff_masked, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200 or area > 20000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            # Classify as stain or dent based on shape
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            d_type = DefectType.DENT if circularity > 0.6 else DefectType.STAIN
            severity = "medium" if area < 2000 else "high"

            defects.append(DefectInfo(
                defect_type=d_type, severity=severity,
                location=(cx, cy), size=int(np.sqrt(area)),
                confidence=min(0.85, 0.55 + area / 10000),
                bbox=(x, y, w, h),
            ))
            color = (255, 165, 0) if d_type == DefectType.DENT else (0, 255, 165)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, d_type.value, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return defects, annotated

    def _compute_anomaly_score(self, defects: List[DefectInfo],
                                mask: np.ndarray, gray: np.ndarray) -> float:
        """Compute overall anomaly score 0-100."""
        if not defects:
            return float(np.random.uniform(0, 8))

        severity_weights = {"low": 5, "medium": 20, "high": 40, "critical": 80}
        score = sum(severity_weights.get(d.severity, 10) * d.confidence for d in defects)
        return min(100.0, score)
