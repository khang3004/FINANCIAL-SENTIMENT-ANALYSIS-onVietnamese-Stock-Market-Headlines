"""
Configuration Management Module

This module handles loading and managing configuration files
for the sentiment analysis pipeline.
"""

from .config_loader import load_config, save_config, Config

__all__ = ['load_config', 'save_config', 'Config']
