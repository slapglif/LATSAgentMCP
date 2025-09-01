"""
Utility helper functions
Contains various code patterns for testing
"""

import re
import json
from typing import Any, List, Optional, Union


def validate_email(email: str) -> bool:
    """Validate email format"""
    # BUG: Overly simple regex
    pattern = r'^[^@]+@[^@]+$'
    return bool(re.match(pattern, email))


def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    # BUG: Doesn't handle all special characters
    dangerous_chars = ['<', '>', '"', "'"]
    for char in dangerous_chars:
        text = text.replace(char, '')
    return text


def parse_config(config_str: str) -> dict:
    """Parse configuration string"""
    try:
        # BUG: No validation of config structure
        return json.loads(config_str)
    except json.JSONDecodeError:
        return {}


class DataProcessor:
    """Process various data formats"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}  # Simple cache
    
    def process_list(self, items: List[Any]) -> List[Any]:
        """Process a list of items"""
        # BUG: No type checking
        result = []
        for item in items[:self.max_size]:
            if item not in self.cache:
                self.cache[item] = self._transform(item)
            result.append(self.cache[item])
        return result
    
    def _transform(self, item: Any) -> Any:
        """Transform an item"""
        if isinstance(item, str):
            return item.upper()
        elif isinstance(item, (int, float)):
            return item * 2
        else:
            # BUG: Returns None for unknown types
            return None
    
    def merge_data(self, data1: dict, data2: dict) -> dict:
        """Merge two dictionaries"""
        # BUG: Shallow merge only
        result = data1.copy()
        result.update(data2)
        return result


def calculate_score(values: List[float], weights: Optional[List[float]] = None) -> float:
    """Calculate weighted score"""
    if not values:
        return 0.0
    
    if weights is None:
        weights = [1.0] * len(values)
    
    # BUG: No check for matching lengths
    total = sum(v * w for v, w in zip(values, weights))
    return total / sum(weights)


def find_pattern(text: str, pattern: str) -> List[int]:
    """Find all occurrences of pattern in text"""
    positions = []
    # BUG: Case-sensitive only
    index = text.find(pattern)
    while index != -1:
        positions.append(index)
        index = text.find(pattern, index + 1)
    return positions


class ConfigManager:
    """Manage application configuration"""
    
    DEFAULT_CONFIG = {
        'debug': False,
        'timeout': 30,
        'retry_count': 3
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, path: str) -> bool:
        """Load configuration from file"""
        try:
            with open(path, 'r') as f:
                # BUG: No validation of loaded config
                self.config.update(json.load(f))
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        # BUG: No nested key support (e.g., 'database.host')
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value