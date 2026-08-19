"""
stubs.py — Explicit Stubs for Non-Achievable Features.

These features CANNOT be trained or recognized with the provided Roboflow dataset
(Vehicles-coco.v2i.multiclass) because no emergency classes or vehicle make/model labels exist.

Do NOT fake, mock, or hardcode plausible outputs.
These functions return explicit unavailable responses and are presented in the UI
as disabled roadmap features with transparent explanations.
"""

from typing import Dict, Any, Optional


def detect_emergency_vehicle(crop_or_frame: Optional[Any] = None) -> Dict[str, Any]:
    # NOT IMPLEMENTED — requires labeled emergency vehicle dataset (Ambulance, Police, Fire Truck)
    return {
        "feature": "Emergency Vehicle Detection",
        "status": "unavailable",
        "available": False,
        "reason": "The provided dataset (Vehicles-coco.v2i.multiclass) only contains labels for [bus, car, motorcycle, truck]. No emergency vehicle class (Ambulance/Police/Fire) is present.",
        "required_data_format": "Images with multi-class or bounding-box annotations including 'ambulance', 'police_car', 'fire_truck' labels in Roboflow/COCO format."
    }


def get_vehicle_model(crop_image: Optional[Any] = None) -> Dict[str, Any]:
    # NOT IMPLEMENTED — requires vehicle make/model fine-grained dataset (e.g. Stanford Cars or CompCars)
    return {
        "feature": "Vehicle Model Recognition",
        "status": "unavailable",
        "available": False,
        "reason": "The provided dataset only classifies general vehicle categories (bus, car, motorcycle, truck). No make/model annotations (e.g., Honda Civic, Toyota Camry, Ford F-150) exist in the training data.",
        "required_data_format": "Fine-grained labeled vehicle dataset with make, model, and year annotations (e.g., Stanford Cars format: make, model, sub-model, year)."
    }


def detect_helmet_violation(motorcycle_crop: Optional[Any] = None) -> Dict[str, Any]:
    # NOT IMPLEMENTED — requires rider & helmet bounding box dataset
    return {
        "feature": "Helmet Violation Detection",
        "status": "unavailable",
        "available": False,
        "reason": "Helmet detection requires object detection annotations for motorcycle riders and helmet/no-helmet classes. The current dataset contains only image-level vehicle labels.",
        "required_data_format": "Bounding box annotated dataset with classes: ['rider_with_helmet', 'rider_without_helmet', 'motorcycle']."
    }
