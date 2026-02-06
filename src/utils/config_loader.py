"""
Configuration Loader Module

Handles loading and accessing YAML configuration files.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


class Config:
    """
    Configuration manager for the AI Surveillance System.
    
    Provides centralized access to all configuration settings
    loaded from YAML files.
    """
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    _rules: Dict[str, Any] = {}
    
    def __new__(cls) -> 'Config':
        """Singleton pattern - only one config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: str, rules_path: Optional[str] = None) -> None:
        """
        Load configuration from YAML files.
        
        Args:
            config_path: Path to main configuration file
            rules_path: Optional path to rules configuration file
        """
        # Load main config
        self._config = self._load_yaml(config_path)
        
        # Load rules config if provided
        if rules_path:
            self._rules = self._load_yaml(rules_path)
        else:
            # Try to find rules config in same directory
            config_dir = Path(config_path).parent
            default_rules_path = config_dir / "rules_config.yaml"
            if default_rules_path.exists():
                self._rules = self._load_yaml(str(default_rules_path))
    
    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """
        Load and parse a YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Parsed configuration dictionary
            
        Raises:
            ConfigError: If file cannot be loaded or parsed
        """
        if not os.path.exists(file_path):
            raise ConfigError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML file {file_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Error reading config file {file_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'video.source_type')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._get_nested(self._config, key, default)
    
    def get_rule(self, key: str, default: Any = None) -> Any:
        """
        Get a rule configuration value using dot notation.
        
        Args:
            key: Rule configuration key
            default: Default value if key not found
            
        Returns:
            Rule configuration value or default
        """
        return self._get_nested(self._rules, key, default)
    
    def _get_nested(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Get a nested value from a dictionary using dot notation.
        
        Args:
            data: Dictionary to search
            key: Dot-separated key path
            default: Default value if not found
            
        Returns:
            Value at key path or default
        """
        keys = key.split('.')
        value = data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def video(self) -> Dict[str, Any]:
        """Get video configuration section."""
        return self._config.get('video', {})
    
    @property
    def models(self) -> Dict[str, Any]:
        """Get models configuration section."""
        return self._config.get('models', {})
    
    @property
    def detection(self) -> Dict[str, Any]:
        """Get detection configuration section."""
        return self._config.get('detection', {})
    
    @property
    def alerts(self) -> Dict[str, Any]:
        """Get alerts configuration section."""
        return self._config.get('alerts', {})
    
    @property
    def ui(self) -> Dict[str, Any]:
        """Get UI configuration section."""
        return self._config.get('ui', {})
    
    @property
    def rules(self) -> Dict[str, Any]:
        """Get all rules configuration."""
        return self._rules


# Global config instance
_config = Config()


def load_config(config_path: str, rules_path: Optional[str] = None) -> Config:
    """
    Load configuration files.
    
    Args:
        config_path: Path to main configuration file
        rules_path: Optional path to rules configuration file
        
    Returns:
        Loaded Config instance
    """
    _config.load(config_path, rules_path)
    return _config


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config instance
    """
    return _config
