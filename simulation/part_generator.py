"""
Synthetic Part & Defect Image Generator
Generates realistic synthetic images of manufactured parts with various defects.
Simulates watchmaking/precision parts (gears, discs, shafts, brackets).

Author: Wassim BELAID
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random


class PartType(Enum):
    GEAR         = "gear"
    DISC         = "disc"
    SHAFT        = "shaft"
    BRACKET      = "bracket"
    BEARING_RING = "bearing_ring"
    WATCH_PART   = "watch_part"


class DefectType(Enum):
    NONE      = "none"
    SCRATCH   = "scratch"
    CRACK     = "crack"
    DENT      = "dent"
    BURR      = "burr"
    PIT       = "pit"
    STAIN     = "stain"
    CHIP      = "chip"
    INCLUSION = "inclusion"


@dataclass
class DefectInfo:
    defect_type: DefectType
    severity: str        # "low", "medium", "high", "critical"
    location: Tuple[int, int]  # (x, y) pixel coordinates
    size: int            # Approximate size in pixels
    confidence: float    # Detection confidence (for simulation)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)

    @property
    def action_required(self) -> str:
        if self.severity == "critical":
            return "REJECT IMMEDIATELY — Do not ship"
        elif self.severity == "high":
            return "REJECT — Return to machining"
        elif self.severity == "medium":
            return "REWORK — Surface treatment required"
        else:
            return "INSPECT — Manual verification recommended"

    @property
    def iso_code(self) -> str:
        codes = {
            DefectType.SCRATCH: "ISO 1302-Ra",
            DefectType.CRACK:   "ISO 6507",
            DefectType.DENT:    "ISO 2768",
            DefectType.BURR:    "ISO 13715",
            DefectType.PIT:     "ISO 4288",
            DefectType.STAIN:   "ISO 8501",
            DefectType.CHIP:    "ISO 286",
            DefectType.INCLUSION: "ASTM E45",
        }
        return codes.get(self.defect_type, "ISO 9001")


class PartGenerator:
    """
    Generates synthetic images of precision manufactured parts with defects.

    Simulates:
    - Base part geometry (gear, disc, shaft, bracket, bearing ring)
    - Realistic surface texture (machined, polished, brushed)
    - Lighting simulation (directional, diffuse)
    - Multiple defect types with realistic appearance
    - Controlled defect injection for testing
    """

    IMAGE_SIZE = (512, 512)
    PART_COLOR_RANGE = ((140, 140, 140), (200, 200, 200))  # Steel/aluminum gray

    SEVERITY_MAP = {
        DefectType.SCRATCH:   ["low", "medium", "high"],
        DefectType.CRACK:     ["high", "critical"],
        DefectType.DENT:      ["medium", "high"],
        DefectType.BURR:      ["low", "medium"],
        DefectType.PIT:       ["medium", "high", "critical"],
        DefectType.STAIN:     ["low", "medium"],
        DefectType.CHIP:      ["medium", "high"],
        DefectType.INCLUSION: ["high", "critical"],
    }

    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate(self, part_type: PartType = None,
                 defect_type: DefectType = None,
                 n_defects: int = None) -> Tuple[np.ndarray, List[DefectInfo]]:
        """
        Generate a synthetic part image with optional defects.

        Args:
            part_type: Type of part (random if None)
            defect_type: Type of defect (random if None)
            n_defects: Number of defects to inject (0-3)

        Returns:
            (image_bgr, list_of_defects)
        """
        if part_type is None:
            part_type = random.choice(list(PartType))
        if n_defects is None:
            n_defects = random.choices([0, 1, 2, 3], weights=[0.35, 0.40, 0.20, 0.05])[0]

        # Generate base part
        img = self._generate_base_part(part_type)
        img = self._add_surface_texture(img)
        img = self._add_lighting(img)

        # Inject defects
        defects = []
        for _ in range(n_defects):
            d_type = defect_type if defect_type else random.choice(
                [d for d in DefectType if d != DefectType.NONE]
            )
            img, defect_info = self._inject_defect(img, d_type, part_type)
            if defect_info:
                defects.append(defect_info)

        # Add camera noise
        img = self._add_camera_noise(img)

        return img, defects

    def _generate_base_part(self, part_type: PartType) -> np.ndarray:
        """Generate base geometry for part type."""
        h, w = self.IMAGE_SIZE
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Background (dark inspection surface)
        bg_color = np.random.randint(20, 45)
        img[:] = bg_color

        cx, cy = w // 2, h // 2
        color = tuple(np.random.randint(c[0], c[1]) for c in
                      zip(self.PART_COLOR_RANGE[0], self.PART_COLOR_RANGE[1]))

        if part_type == PartType.GEAR:
            # Draw gear with teeth
            r_outer = 200
            r_inner = 150
            n_teeth = 24
            pts = []
            for i in range(n_teeth * 2):
                angle = i * np.pi / n_teeth
                r = r_outer if i % 2 == 0 else r_inner
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                pts.append([x, y])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(img, [pts], color)
            # Center bore
            cv2.circle(img, (cx, cy), 40, (bg_color, bg_color, bg_color), -1)
            # Keyway
            cv2.rectangle(img, (cx - 8, cy - 45), (cx + 8, cy + 45), (bg_color, bg_color, bg_color), -1)

        elif part_type == PartType.DISC:
            cv2.circle(img, (cx, cy), 220, color, -1)
            cv2.circle(img, (cx, cy), 50, (bg_color, bg_color, bg_color), -1)
            # Bolt holes
            for i in range(6):
                angle = i * np.pi / 3
                bx = int(cx + 160 * np.cos(angle))
                by = int(cy + 160 * np.sin(angle))
                cv2.circle(img, (bx, by), 15, (bg_color, bg_color, bg_color), -1)

        elif part_type == PartType.SHAFT:
            # Cylindrical shaft (front view = rectangle)
            cv2.rectangle(img, (cx - 60, 50), (cx + 60, h - 50), color, -1)
            # Chamfers
            cv2.circle(img, (cx, 70), 62, color, -1)
            cv2.circle(img, (cx, h - 70), 62, color, -1)
            # Keyway
            cv2.rectangle(img, (cx - 15, cy - 80), (cx + 15, cy + 80), (bg_color, bg_color, bg_color), -1)

        elif part_type == PartType.BRACKET:
            # L-shaped bracket
            pts = np.array([[50, 50], [460, 50], [460, 180], [200, 180],
                           [200, 460], [50, 460]], np.int32)
            cv2.fillPoly(img, [pts], color)
            # Mounting holes
            for pos in [(130, 100), (360, 100), (100, 350)]:
                cv2.circle(img, pos, 22, (bg_color, bg_color, bg_color), -1)

        elif part_type == PartType.BEARING_RING:
            cv2.circle(img, (cx, cy), 220, color, -1)
            cv2.circle(img, (cx, cy), 160, (bg_color, bg_color, bg_color), -1)
            # Race grooves
            cv2.circle(img, (cx, cy), 190, (max(0, color[0] - 20),) * 3, 3)
            cv2.circle(img, (cx, cy), 170, (max(0, color[0] - 20),) * 3, 3)

        elif part_type == PartType.WATCH_PART:
            # Watch movement plate
            cv2.ellipse(img, (cx, cy), (200, 180), 0, 0, 360, color, -1)
            # Jewel holes
            for pos in [(cx-80, cy-60), (cx+80, cy-60), (cx, cy+80),
                        (cx-50, cy+20), (cx+50, cy+20)]:
                cv2.circle(img, pos, 8, (bg_color, bg_color, bg_color), -1)
                # Jewel (red)
                cv2.circle(img, pos, 5, (0, 0, 180), -1)
            # Bridges
            for angle in range(0, 360, 45):
                r = np.radians(angle)
                x1 = int(cx + 140 * np.cos(r))
                y1 = int(cy + 140 * np.sin(r))
                cv2.circle(img, (x1, y1), 12, (bg_color, bg_color, bg_color), -1)

        return img

    def _add_surface_texture(self, img: np.ndarray) -> np.ndarray:
        """Add realistic machined surface texture."""
        texture = np.random.normal(0, 3, img.shape).astype(np.int16)
        # Directional machining marks (turning/milling)
        scratch_layer = np.zeros(img.shape[:2], dtype=np.uint8)
        for _ in range(random.randint(20, 60)):
            angle = random.uniform(0, np.pi)
            x = random.randint(0, img.shape[1])
            y = random.randint(0, img.shape[0])
            length = random.randint(30, 150)
            dx = int(length * np.cos(angle))
            dy = int(length * np.sin(angle))
            cv2.line(scratch_layer, (x, y), (x+dx, y+dy), 8, 1)
        scratch_3ch = np.stack([scratch_layer]*3, axis=-1).astype(np.int16)
        texture = texture - scratch_3ch

        result = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        return result

    def _add_lighting(self, img: np.ndarray) -> np.ndarray:
        """Add directional lighting effect."""
        h, w = img.shape[:2]
        # Gradient lighting
        light_dir = random.choice([(1, 0), (0, 1), (0.7, 0.7)])
        x_grad = np.linspace(0, 1, w) * light_dir[0]
        y_grad = np.linspace(0, 1, h)[:, np.newaxis] * light_dir[1]
        gradient = (x_grad + y_grad) / 2
        gradient = (gradient * 30 - 15).astype(np.int16)

        for c in range(3):
            img[:, :, c] = np.clip(
                img[:, :, c].astype(np.int16) + gradient, 0, 255
            ).astype(np.uint8)

        return img

    def _inject_defect(self, img: np.ndarray, defect_type: DefectType,
                        part_type: PartType) -> Tuple[np.ndarray, Optional[DefectInfo]]:
        """Inject a realistic defect into the image."""
        h, w = img.shape[:2]

        # Find part pixels (not background)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        part_mask = gray > 50
        part_pixels = np.argwhere(part_mask)

        if len(part_pixels) == 0:
            return img, None

        # Random location on part
        idx = random.randint(0, len(part_pixels) - 1)
        y, x = part_pixels[idx]

        severity_options = self.SEVERITY_MAP.get(defect_type, ["medium"])
        severity = random.choice(severity_options)

        # Size based on severity
        size_map = {"low": (5, 15), "medium": (15, 30), "high": (30, 50), "critical": (50, 80)}
        size = random.randint(*size_map.get(severity, (10, 25)))

        defect_color_dark = tuple(max(0, c - random.randint(40, 80)) for c in (160, 160, 160))
        defect_color_light = tuple(min(255, c + random.randint(30, 60)) for c in (160, 160, 160))

        if defect_type == DefectType.SCRATCH:
            angle = random.uniform(0, np.pi)
            length = size * random.randint(3, 8)
            dx, dy = int(length * np.cos(angle)), int(length * np.sin(angle))
            thickness = random.randint(1, 3)
            cv2.line(img, (x, y), (x + dx, y + dy), defect_color_dark, thickness)
            # Highlight reflection
            cv2.line(img, (x+1, y+1), (x+dx+1, y+dy+1), defect_color_light, 1)

        elif defect_type == DefectType.CRACK:
            # Branching crack
            def draw_crack(img, x, y, angle, length, depth=0):
                if depth > 3 or length < 5:
                    return
                dx = int(length * np.cos(angle))
                dy = int(length * np.sin(angle))
                x2, y2 = x + dx, y + dy
                cv2.line(img, (x, y), (x2, y2), (0, 0, 0), random.randint(1, 3))
                # Branch
                if random.random() > 0.4:
                    branch_angle = angle + random.uniform(0.3, 0.8)
                    draw_crack(img, x2, y2, branch_angle, length * 0.6, depth + 1)
                draw_crack(img, x2, y2, angle + random.uniform(-0.2, 0.2),
                          length * 0.7, depth + 1)
            draw_crack(img, x, y, random.uniform(0, np.pi), size * 2)

        elif defect_type == DefectType.DENT:
            # Circular depression with shadow
            cv2.circle(img, (x, y), size, defect_color_dark, -1)
            cv2.circle(img, (x - size//4, y - size//4), size - 3, defect_color_light, 2)

        elif defect_type == DefectType.BURR:
            # Irregular protrusion
            pts = []
            for i in range(8):
                angle = i * np.pi / 4
                r = size + random.randint(-5, 10)
                pts.append([x + int(r * np.cos(angle)), y + int(r * np.sin(angle))])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(img, [pts], defect_color_light)
            cv2.polylines(img, [pts], True, (200, 200, 200), 1)

        elif defect_type == DefectType.PIT:
            # Deep circular pit
            cv2.circle(img, (x, y), size, (15, 15, 15), -1)
            cv2.circle(img, (x + size//3, y - size//3), size//3, (60, 60, 60), -1)

        elif defect_type == DefectType.STAIN:
            # Irregular stain blob
            stain_color = (random.randint(80, 130), random.randint(80, 130),
                          random.randint(60, 110))
            for _ in range(random.randint(5, 15)):
                ox, oy = random.randint(-size, size), random.randint(-size, size)
                r = random.randint(size // 3, size)
                cv2.circle(img, (x + ox, y + oy), r, stain_color, -1)
            # Blur for realistic look
            roi = img[max(0, y-size*2):y+size*2, max(0, x-size*2):x+size*2]
            if roi.size > 0:
                img[max(0, y-size*2):y+size*2, max(0, x-size*2):x+size*2] = cv2.GaussianBlur(roi, (7, 7), 2)

        elif defect_type == DefectType.CHIP:
            # Missing material chunk
            pts = []
            for i in range(6):
                angle = i * np.pi / 3
                r = size + random.randint(-size//3, size//3)
                pts.append([x + int(r * np.cos(angle)), y + int(r * np.sin(angle))])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(img, [pts], (int(int(gray[y, x])) // 2, int(int(gray[y, x])) // 2, int(int(gray[y, x])) // 2))

        elif defect_type == DefectType.INCLUSION:
            # Dark foreign material inclusion
            inclusion_color = (random.randint(0, 40), random.randint(0, 40),
                              random.randint(0, 60))
            cv2.ellipse(img, (x, y), (size, size // 2),
                       random.randint(0, 180), 0, 360, inclusion_color, -1)

        # Bounding box
        bbox = (max(0, x - size), max(0, y - size),
                min(size * 2, w - x), min(size * 2, h - y))

        defect_info = DefectInfo(
            defect_type=defect_type,
            severity=severity,
            location=(x, y),
            size=size,
            confidence=random.uniform(0.75, 0.99),
            bbox=bbox,
        )

        return img, defect_info

    def _add_camera_noise(self, img: np.ndarray) -> np.ndarray:
        """Add realistic camera sensor noise."""
        noise = np.random.normal(0, 2, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def generate_batch(self, n: int = 50, defect_rate: float = 0.4) -> List[Dict]:
        """Generate a batch of parts for inspection."""
        batch = []
        part_types = list(PartType)
        for i in range(n):
            part_type = random.choice(part_types)
            has_defect = random.random() < defect_rate
            n_def = random.randint(1, 2) if has_defect else 0
            img, defects = self.generate(part_type=part_type, n_defects=n_def)
            batch.append({
                "part_id": f"P{i+1:04d}",
                "part_type": part_type.value,
                "image": img,
                "defects": defects,
                "has_defect": len(defects) > 0,
                "n_defects": len(defects),
            })
        return batch
