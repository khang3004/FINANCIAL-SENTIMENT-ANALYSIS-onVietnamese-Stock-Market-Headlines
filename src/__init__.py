"""
Main Source Package for Vietnamese Financial Sentiment Analysis

This package provides comprehensive tools for:
- Text preprocessing and tokenization
- Model training and evaluation
- Sentiment prediction
- Data processing pipelines
"""

__version__ = "1.0.0"
__author__ = "Vietnamese Financial Sentiment Analysis Team"

from . import data_processing
from . import utils

__all__ = [
    'data_processing',
    'utils',
]
