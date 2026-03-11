"""CCPD ALPR package."""

from .service import ALPRService, ALPRPrediction
from .province_classifier import ProvinceClassifier
from .province_refiner import ProvinceRefiner

__all__ = ["ALPRService", "ALPRPrediction", "ProvinceRefiner", "ProvinceClassifier"]
