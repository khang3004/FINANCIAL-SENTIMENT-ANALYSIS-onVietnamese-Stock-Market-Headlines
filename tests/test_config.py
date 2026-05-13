"""
Tests for Configuration Loading

This module contains unit tests for the configuration management system.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.utils.config_loader import (
    load_config, 
    save_config, 
    Config,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    get_default_config
)


class TestConfigClasses:
    """Test configuration data classes."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.name == "phobert"
        assert config.max_length == 125
        assert config.batch_size == 32
        assert config.epochs == 10
        assert config.learning_rate == 2e-5
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        assert "datacw.xlsx" in config.train_path
        assert "out_of_sample" in config.test_path
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.seed == 42
        assert config.device == "cuda"
        assert config.early_stopping is True
        assert config.patience == 3


class TestMainConfig:
    """Test main Config class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
    
    def test_config_initialization(self):
        """Test Config initialization with defaults."""
        assert isinstance(self.config.model, ModelConfig)
        assert isinstance(self.config.data, DataConfig)
        assert isinstance(self.config.training, TrainingConfig)
    
    def test_config_to_dict(self):
        """Test converting Config to dictionary."""
        config_dict = self.config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'data' in config_dict
        assert 'training' in config_dict
    
    def test_config_from_dict(self):
        """Test creating Config from dictionary."""
        test_dict = {
            'model': {
                'name': 'lstm',
                'max_length': 256,
                'batch_size': 64,
                'epochs': 20,
                'learning_rate': 0.001,
            },
            'data': {
                'train_path': 'custom/train.xlsx',
                'test_path': 'custom/test.xlsx',
            },
            'training': {
                'seed': 123,
                'device': 'cpu',
            }
        }
        config = Config.from_dict(test_dict)
        assert config.model.name == 'lstm'
        assert config.model.max_length == 256
        assert config.training.seed == 123
        assert config.training.device == 'cpu'


class TestConfigFileOperations:
    """Test configuration file operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
    
    def teardown_method(self):
        """Cleanup temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Modify config
        self.config.model.name = 'lstm'
        self.config.model.epochs = 15
        
        # Save config
        save_config(self.config, config_path)
        
        # Verify file exists
        assert os.path.exists(config_path)
        
        # Load config
        loaded_config = load_config(config_path)
        
        # Verify values
        assert loaded_config.model.name == 'lstm'
        assert loaded_config.model.epochs == 15
    
    def test_load_nonexistent_config(self):
        """Test loading nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.yaml')
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        assert isinstance(config, Config)
        assert config.model.name == 'phobert'


class TestConfigEdgeCases:
    """Test edge cases in configuration."""
    
    def test_partial_dict_loading(self):
        """Test loading config with partial dictionary."""
        partial_dict = {
            'model': {
                'name': 'custom_model'
            }
        }
        config = Config.from_dict(partial_dict)
        assert config.model.name == 'custom_model'
        # Other values should be defaults
        assert config.model.max_length == 125
    
    def test_empty_dict_loading(self):
        """Test loading config with empty dictionary."""
        config = Config.from_dict({})
        assert config.model.name == 'phobert'  # Default value
    
    def test_invalid_yaml_content(self):
        """Test handling invalid YAML content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(Exception):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
