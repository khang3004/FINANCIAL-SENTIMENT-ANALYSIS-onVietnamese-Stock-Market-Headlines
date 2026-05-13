"""
Configuration Loader and Manager

This module provides functionality to load, save, and manage
configuration files in YAML format.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    name: str = "phobert"
    max_length: int = 125
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 2e-5
    num_labels: int = 2


@dataclass
class DataConfig:
    """Configuration for data paths and settings."""
    train_path: str = "data/data_2/datacw.xlsx"
    test_path: str = "data/data_2/out_of_sample_data.xlsx"
    raw_news_path: str = "data/data_1/media07_01_2023.xlsx"
    raw_price_path: str = "data/data_1/price07_01_2023.xlsx"


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    seed: int = 42
    device: str = "cuda"
    early_stopping: bool = True
    patience: int = 3
    warmup_steps: int = 1000
    weight_decay: float = 0.01


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': {
                'name': self.model.name,
                'max_length': self.model.max_length,
                'batch_size': self.model.batch_size,
                'epochs': self.model.epochs,
                'learning_rate': self.model.learning_rate,
                'num_labels': self.model.num_labels,
            },
            'data': {
                'train_path': self.data.train_path,
                'test_path': self.data.test_path,
                'raw_news_path': self.data.raw_news_path,
                'raw_price_path': self.data.raw_price_path,
            },
            'training': {
                'seed': self.training.seed,
                'device': self.training.device,
                'early_stopping': self.training.early_stopping,
                'patience': self.training.patience,
                'warmup_steps': self.training.warmup_steps,
                'weight_decay': self.training.weight_decay,
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        config = cls()
        
        if 'model' in config_dict:
            model_data = config_dict['model']
            config.model = ModelConfig(
                name=model_data.get('name', 'phobert'),
                max_length=model_data.get('max_length', 125),
                batch_size=model_data.get('batch_size', 32),
                epochs=model_data.get('epochs', 10),
                learning_rate=model_data.get('learning_rate', 2e-5),
                num_labels=model_data.get('num_labels', 2),
            )
        
        if 'data' in config_dict:
            data_data = config_dict['data']
            config.data = DataConfig(
                train_path=data_data.get('train_path', 'data/data_2/datacw.xlsx'),
                test_path=data_data.get('test_path', 'data/data_2/out_of_sample_data.xlsx'),
                raw_news_path=data_data.get('raw_news_path', 'data/data_1/media07_01_2023.xlsx'),
                raw_price_path=data_data.get('raw_price_path', 'data/data_1/price07_01_2023.xlsx'),
            )
        
        if 'training' in config_dict:
            training_data = config_dict['training']
            config.training = TrainingConfig(
                seed=training_data.get('seed', 42),
                device=training_data.get('device', 'cuda'),
                early_stopping=training_data.get('early_stopping', True),
                patience=training_data.get('patience', 3),
                warmup_steps=training_data.get('warmup_steps', 1000),
                weight_decay=training_data.get('weight_decay', 0.01),
            )
        
        return config


def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config object with loaded settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Config object to save
        config_path: Path to save the YAML file
        
    Raises:
        IOError: If file cannot be written
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)


def get_default_config() -> Config:
    """
    Get default configuration.
    
    Returns:
        Config object with default settings
    """
    return Config()
