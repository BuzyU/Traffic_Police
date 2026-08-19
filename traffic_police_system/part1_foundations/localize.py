"""
localize.py — Classical CV Localization + Classification Pipeline.

Because the dataset has NO bounding box labels, localization is done purely
with classical computer vision:
  - Video: MOG2 background subtraction + contour filtering + NMS
  - Stills: Canny edges + morphological closing + contour filtering + NMS

Each localized region is cropped, passed through the trained classifier,
and assigned a class label and corresponding bounding box color:
  - truck      → RED     (BGR: 30, 30, 220)
  - motorcycle → BLUE    (BGR: 255, 100, 30)
  - bus        → BLACK   (BGR: 0, 0, 0)
  - car        → YELLOW  (BGR: 0, 220, 255)
"""

import sys
from pathlib import Path

SITE_PKGS = Path(__file__).parent.parent / "site_pkgs"
if SITE_PKGS.exists() and str(SITE_PKGS) not in sys.path:
    sys.path.insert(0, str(SITE_PKGS))

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Color mapping in BGR for OpenCV drawing
CLASS_COLORS_BGR = {
    "truck":      (30, 30, 220),    # Red
    "motorcycle": (255, 100, 30),   # Blue
    "bus":        (0, 0, 0),        # Black
    "car":        (0, 220, 255),    # Yellow
}

CLASSES = ["bus", "car", "motorcycle", "truck"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

crop_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def non_max_suppression_boxes(boxes: np.ndarray, overlap_thresh: float = 0.3) -> np.ndarray:
    """NMS algorithm to eliminate overlapping bounding boxes."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=int)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")


class ClassicalLocalizer:
    def __init__(self, mode: str = "video", min_area: int = 1500, max_area: int = 250000):
        self.mode = mode
        self.min_area = min_area
        self.max_area = max_area
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )

    def extract_candidate_regions(self, frame: np.ndarray) -> np.ndarray:
        """Extract candidate bounding boxes (x, y, w, h) using non-ML classical CV."""
        h_frame, w_frame = frame.shape[:2]
        boxes = []

        if self.mode == "video":
            # Background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            # Remove noise via morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(fg_mask, kernel, iterations=2)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Still image edge/blob detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
            dilated = cv2.dilate(closed, kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 0.25 <= aspect_ratio <= 4.0:
                    boxes.append([x, y, w, h])

        if len(boxes) == 0:
            return np.empty((0, 4), dtype=int)

        return non_max_suppression_boxes(np.array(boxes), overlap_thresh=0.3)


class VehiclePipeline:
    def __init__(self, model: Optional[nn.Module] = None, device: Optional[torch.device] = None, mode: str = "video"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        self.localizer = ClassicalLocalizer(mode=mode)

    @torch.no_grad()
    def classify_crop(self, crop: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Classify cropped vehicle image with the trained model."""
        if self.model is None:
            return "car", 1.0, {"car": 1.0, "bus": 0.0, "motorcycle": 0.0, "truck": 0.0}

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_crop)
        tensor = crop_transform(pil_img).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

        prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        best_idx = int(np.argmax(probs))
        best_class = CLASSES[best_idx]
        best_prob = float(probs[best_idx])

        return best_class, best_prob, prob_dict

    def process_frame(self, frame: np.ndarray, conf_thresh: float = 0.3) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
        """
        Runs localization + classification, overlays boxes & counts.
        Returns:
            annotated_frame: np.ndarray
            detections: List of dicts {box: [x,y,w,h], class: str, conf: float, color: tuple}
            counts: Dict of {class: count}
        """
        candidate_boxes = self.localizer.extract_candidate_regions(frame)
        annotated = frame.copy()
        detections = []
        counts = {c: 0 for c in CLASSES}

        for (x, y, w, h) in candidate_boxes:
            crop = frame[y:y+h, x:x+w]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            cls_name, conf, _ = self.classify_crop(crop)
            if conf >= conf_thresh:
                counts[cls_name] += 1
                color = CLASS_COLORS_BGR.get(cls_name, (0, 255, 0))

                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

                # Draw label badge with background
                label = f"{cls_name.capitalize()} {conf:.2f}"
                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x, y - lh - 6), (x + lw + 6, y), color, -1)
                
                # Text color: white if box is dark, black if bright
                text_color = (255, 255, 255) if cls_name in ["bus", "motorcycle"] else (0, 0, 0)
                cv2.putText(annotated, label, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

                detections.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "class": cls_name,
                    "conf": float(conf),
                    "color": color
                })

        # Draw running count overlay panel
        self._draw_stats_overlay(annotated, counts)

        return annotated, detections, counts

    def _draw_stats_overlay(self, frame: np.ndarray, counts: Dict[str, int]):
        """Draw a dashboard legend & running count overlay on the frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Top-left HUD card
        cv2.rectangle(overlay, (10, 10), (220, 140), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "REAL-TIME VEHICLE COUNTS", (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        y_offset = 50
        for cls_name in ["car", "motorcycle", "bus", "truck"]:
            color = CLASS_COLORS_BGR[cls_name]
            cnt = counts[cls_name]
            # Color badge
            cv2.rectangle(frame, (18, y_offset - 10), (32, y_offset + 2), color, -1)
            cv2.rectangle(frame, (18, y_offset - 10), (32, y_offset + 2), (255, 255, 255), 1)
            # Text
            text = f"{cls_name.capitalize()}: {cnt}"
            cv2.putText(frame, text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            y_offset += 22
