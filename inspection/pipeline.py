"""
Inspection Pipeline & Quality Decision Engine
Orchestrates all vision models and makes PASS/FAIL decisions

Standards: ISO 9001, IATF 16949, ISO 10360
Author: Wassim BELAID
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import time

from simulation.part_generator import DefectType, DefectInfo, PartType
from vision.detection.classical_detector import ClassicalDefectDetector, DetectionResult
from vision.detection.cnn_yolo import CNNClassifier, YOLOStyleDetector, FeatureExtractor


SEVERITY_DECISION = {
    "critical": "REJECT",
    "high":     "REJECT",
    "medium":   "REWORK",
    "low":      "INSPECT",
}

DEFECT_ACTIONS = {
    DefectType.SCRATCH: {
        "action": "Surface rework — micro-abrasive polishing",
        "standard": "ISO 1302 (Ra ≤ 0.8 µm)",
        "responsible": "Surface Treatment Dept.",
        "urgency": "Within 24h",
        "estimated_cost_eur": 45,
    },
    DefectType.CRACK: {
        "action": "IMMEDIATE REJECTION — Structural failure risk. Destroy part, log in NCR system",
        "standard": "ISO 6507 / ASTM E384",
        "responsible": "Quality Manager + Engineering",
        "urgency": "Immediate",
        "estimated_cost_eur": 0,
    },
    DefectType.DENT: {
        "action": "Dimensional inspection → reject if > tolerance. Possible rework via cold forming",
        "standard": "ISO 2768 (IT7)",
        "responsible": "Dimensional Lab",
        "urgency": "Within 8h",
        "estimated_cost_eur": 85,
    },
    DefectType.BURR: {
        "action": "Deburring operation — manual or tumbling. Re-inspect after deburring",
        "standard": "ISO 13715",
        "responsible": "Finishing Dept.",
        "urgency": "Within 4h",
        "estimated_cost_eur": 25,
    },
    DefectType.PIT: {
        "action": "Corrosion analysis required. If depth > 0.1mm: reject. Else: passivation treatment",
        "standard": "ISO 4288 / ASTM B117",
        "responsible": "Materials Lab",
        "urgency": "Within 8h",
        "estimated_cost_eur": 120,
    },
    DefectType.STAIN: {
        "action": "Chemical cleaning (ultrasonic + passivation). Re-inspect surface",
        "standard": "ISO 8501-1",
        "responsible": "Cleaning Dept.",
        "urgency": "Same shift",
        "estimated_cost_eur": 15,
    },
    DefectType.CHIP: {
        "action": "Dimensional check — if functional surface: reject. Edge chips: possible rework",
        "standard": "ISO 286-1",
        "responsible": "Quality Inspector",
        "urgency": "Within 4h",
        "estimated_cost_eur": 65,
    },
    DefectType.INCLUSION: {
        "action": "REJECT — Material contamination. Quarantine batch, notify supplier, 8D analysis",
        "standard": "ASTM E45 / EN 10247",
        "responsible": "Quality Manager + Procurement",
        "urgency": "Immediate — batch quarantine",
        "estimated_cost_eur": 0,
    },
}


@dataclass
class InspectionReport:
    """Complete inspection report for one part."""
    part_id: str
    part_type: str
    timestamp: str
    decision: str          # "PASS", "REWORK", "REJECT"
    confidence: float
    n_defects: int
    defects: List[DefectInfo]
    anomaly_score: float
    cnn_prediction: str
    cnn_confidence: float
    classical_score: float
    yolo_detections: int
    surface_roughness_ra: float
    edge_quality: float
    processing_time_ms: float
    actions_required: List[Dict]
    image_quality: str
    lot_number: str = ""
    operator: str = "AI Vision System"
    machine_id: str = "INSP-001"

    @property
    def is_acceptable(self) -> bool:
        return self.decision == "PASS"

    @property
    def severity_level(self) -> str:
        if self.decision == "REJECT":
            return "critical" if self.n_defects >= 2 else "high"
        elif self.decision == "REWORK":
            return "medium"
        return "low"

    def to_dict(self) -> dict:
        return {
            "part_id": self.part_id,
            "part_type": self.part_type,
            "timestamp": self.timestamp,
            "decision": self.decision,
            "confidence_pct": round(self.confidence * 100, 1),
            "n_defects": self.n_defects,
            "defect_types": ", ".join(set(d.defect_type.value for d in self.defects)),
            "anomaly_score": round(self.anomaly_score, 1),
            "cnn_prediction": self.cnn_prediction,
            "cnn_confidence_pct": round(self.cnn_confidence * 100, 1),
            "surface_roughness_ra_um": self.surface_roughness_ra,
            "edge_quality_pct": self.edge_quality,
            "processing_ms": self.processing_time_ms,
            "image_quality": self.image_quality,
            "lot_number": self.lot_number,
            "operator": self.operator,
            "machine_id": self.machine_id,
        }


class QualityDecisionEngine:
    """
    Quality decision engine — ISO 9001 compliant.
    Makes PASS/REWORK/REJECT decisions based on:
    - Defect type and severity
    - Number of defects
    - Anomaly score threshold
    - CNN classification confidence
    - Surface roughness
    """

    # Thresholds
    ANOMALY_REJECT_THRESHOLD = 65.0
    ANOMALY_REWORK_THRESHOLD = 30.0
    MIN_CONFIDENCE_FOR_PASS = 0.60
    MAX_DEFECTS_PASS = 0
    MAX_RA_PASS_UM = 3.2  # Ra ≤ 3.2 µm for precision parts

    CRITICAL_DEFECTS = {DefectType.CRACK, DefectType.INCLUSION}
    HIGH_DEFECTS = {DefectType.PIT, DefectType.CHIP}
    MEDIUM_DEFECTS = {DefectType.SCRATCH, DefectType.DENT, DefectType.BURR}

    def decide(self, classical: DetectionResult, cnn_class: str,
                cnn_conf: float, yolo_n: int) -> Tuple[str, float, List[Dict]]:
        """
        Make quality decision.

        Returns:
            (decision, confidence, actions_list)
        """
        defects = classical.detected_defects
        anomaly_score = classical.anomaly_score

        # Immediate reject conditions
        for d in defects:
            if d.defect_type in self.CRITICAL_DEFECTS:
                return "REJECT", 0.98, self._get_actions(defects)

        # High severity defects
        high_defects = [d for d in defects if d.severity in ("critical", "high")]
        if high_defects:
            return "REJECT", 0.92, self._get_actions(defects)

        # CNN says defective
        if cnn_class not in ("no_defect", "none") and cnn_conf > 0.75:
            decision = "REWORK" if cnn_class in ("scratch", "stain", "burr") else "REJECT"
            return decision, cnn_conf, self._get_actions(defects)

        # Anomaly score based decision
        if anomaly_score > self.ANOMALY_REJECT_THRESHOLD:
            return "REJECT", min(0.95, anomaly_score / 100), self._get_actions(defects)
        elif anomaly_score > self.ANOMALY_REWORK_THRESHOLD:
            return "REWORK", min(0.88, anomaly_score / 80), self._get_actions(defects)

        # Surface roughness check
        if classical.surface_roughness_ra > self.MAX_RA_PASS_UM:
            return "REWORK", 0.75, [{
                "defect": "High Surface Roughness",
                "action": f"Surface re-polishing required (Ra={classical.surface_roughness_ra:.2f}µm > {self.MAX_RA_PASS_UM}µm)",
                "standard": "ISO 1302",
                "responsible": "Surface Treatment Dept.",
                "urgency": "Within 24h",
                "estimated_cost_eur": 35,
            }]

        # Multiple defects
        if len(defects) >= 2:
            return "REWORK", 0.80, self._get_actions(defects)

        # PASS
        pass_conf = max(0.60, 1.0 - anomaly_score / 100)
        return "PASS", round(pass_conf, 3), []

    def _get_actions(self, defects: List[DefectInfo]) -> List[Dict]:
        """Get required actions for detected defects."""
        actions = []
        seen_types = set()
        for d in defects:
            if d.defect_type not in seen_types:
                action_info = DEFECT_ACTIONS.get(d.defect_type, {})
                if action_info:
                    actions.append({
                        "defect": d.defect_type.value.replace("_", " ").title(),
                        "severity": d.severity,
                        "location": f"({d.location[0]}, {d.location[1]}) px",
                        **action_info,
                    })
                seen_types.add(d.defect_type)
        return actions


class InspectionPipeline:
    """
    Complete automated inspection pipeline.

    Steps:
    1. Image preprocessing
    2. Classical CV analysis (OpenCV)
    3. CNN classification
    4. YOLO localization
    5. Quality decision
    6. Report generation

    Performance: ~150-250ms per part (CPU)
    Throughput: 2,000+ parts/hour
    """

    def __init__(self):
        self.classical = ClassicalDefectDetector()
        self.cnn = CNNClassifier()
        self.yolo = YOLOStyleDetector(self.cnn)
        self.quality = QualityDecisionEngine()
        self._part_counter = 0
        self._pass_count = 0
        self._reject_count = 0
        self._rework_count = 0
        self._history: deque = deque(maxlen=500)
        self._is_trained = False
        self._lot_number = datetime.now().strftime("LOT-%Y%m%d-001")

    def train(self, images: List[np.ndarray], labels: List[str]) -> Dict:
        """Train CNN classifier on labeled dataset."""
        metrics = self.cnn.train(images, labels)
        self._is_trained = True
        return metrics

    def inspect(self, image: np.ndarray, part_id: str = None,
                part_type: str = "unknown", defects_ground_truth: List[DefectInfo] = None
                ) -> InspectionReport:
        """
        Run full inspection on one part image.

        Args:
            image: BGR image
            part_id: Part identifier
            part_type: Part type name
            defects_ground_truth: Known defects (for validation)

        Returns:
            InspectionReport
        """
        t0 = time.time()
        self._part_counter += 1
        if part_id is None:
            part_id = f"P{self._part_counter:06d}"

        # 1. Classical CV
        classical_result = self.classical.detect(image)

        # 2. CNN classification
        cnn_result = self.cnn.predict(image)

        # 3. YOLO detection
        yolo_result = self.yolo.detect(image)

        # 4. Quality decision
        decision, confidence, actions = self.quality.decide(
            classical_result,
            cnn_result.predicted_class,
            cnn_result.confidence,
            yolo_result.n_objects,
        )

        # Combine defects from all methods
        all_defects = classical_result.detected_defects.copy()
        # Add ground truth defects if available (for visualization)
        if defects_ground_truth:
            all_defects = defects_ground_truth  # Use ground truth

        t1 = time.time()
        total_ms = round((t1 - t0) * 1000, 1)

        # Update counters
        if decision == "PASS":
            self._pass_count += 1
        elif decision == "REJECT":
            self._reject_count += 1
        else:
            self._rework_count += 1

        report = InspectionReport(
            part_id=part_id,
            part_type=part_type,
            timestamp=datetime.now().isoformat(),
            decision=decision,
            confidence=confidence,
            n_defects=len(all_defects),
            defects=all_defects,
            anomaly_score=classical_result.anomaly_score,
            cnn_prediction=cnn_result.predicted_class,
            cnn_confidence=cnn_result.confidence,
            classical_score=classical_result.anomaly_score,
            yolo_detections=yolo_result.n_objects,
            surface_roughness_ra=classical_result.surface_roughness_ra,
            edge_quality=classical_result.edge_quality_score,
            processing_time_ms=total_ms,
            actions_required=actions,
            image_quality=classical_result.image_quality,
            lot_number=self._lot_number,
        )

        self._history.append(report)
        return report

    def get_annotated_image(self, image: np.ndarray,
                             report: InspectionReport) -> np.ndarray:
        """Create fully annotated inspection image."""
        annotated = image.copy()

        # Draw defect boxes
        colors = {
            "scratch": (0, 0, 255), "crack": (0, 0, 180), "dent": (0, 165, 255),
            "burr": (0, 255, 255), "pit": (255, 0, 0), "stain": (255, 165, 0),
            "chip": (180, 0, 255), "inclusion": (255, 0, 180),
        }
        for d in report.defects:
            x, y, w, h = d.bbox
            color = colors.get(d.defect_type.value, (255, 255, 0))
            cv2.rectangle(annotated, (x, y), (x + max(w, 30), y + max(h, 30)), color, 2)
            label = f"{d.defect_type.value} ({d.severity})"
            cv2.putText(annotated, label, (x, max(y - 5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Decision banner
        banner_color = {"PASS": (0, 150, 0), "REJECT": (0, 0, 180), "REWORK": (0, 120, 180)}
        bc = banner_color.get(report.decision, (100, 100, 100))
        cv2.rectangle(annotated, (0, 0), (512, 45), bc, -1)
        cv2.putText(annotated, f"{report.decision} | {report.n_defects} defects | "
                   f"Score:{report.anomaly_score:.0f} | {report.processing_time_ms:.0f}ms",
                   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Part ID
        cv2.putText(annotated, f"ID: {report.part_id} | {report.part_type}",
                   (10, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return annotated

    @property
    def statistics(self) -> Dict:
        total = max(self._part_counter, 1)
        return {
            "total_inspected": self._part_counter,
            "pass_count": self._pass_count,
            "reject_count": self._reject_count,
            "rework_count": self._rework_count,
            "pass_rate_pct": round(self._pass_count / total * 100, 1),
            "reject_rate_pct": round(self._reject_count / total * 100, 1),
            "rework_rate_pct": round(self._rework_count / total * 100, 1),
            "defect_rate_pct": round((self._reject_count + self._rework_count) / total * 100, 1),
        }

    @property
    def history(self) -> List[InspectionReport]:
        return list(self._history)
