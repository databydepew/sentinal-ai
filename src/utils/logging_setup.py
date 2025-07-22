import logging
from typing import Dict, Any

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure handlers if they haven't been set up already
    if not logger.handlers:
        # Set default level if not set
        if not logger.level:
            logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class ContextualLogger:
    def __init__(self, base_logger: logging.Logger, context: Dict[str, Any]):
        self.base_logger = base_logger
        self.context = context
    
    def _format_message(self, message: str) -> str:
        if not self.context:
            return message
        context_str = ", ".join(f"{key}={value}" for key, value in self.context.items())
        return f"[{context_str}] {message}"
    
    def debug(self, message: str, **kwargs) -> None:
        self.base_logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        self.base_logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self.base_logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self.base_logger.error(self._format_message(message), **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        self.base_logger.exception(self._format_message(message), **kwargs)
    
    def add_context(self, **kwargs) -> 'ContextualLogger':
        """Add additional context and return new logger"""
        new_context = self.context.copy()
        new_context.update(kwargs)
        return ContextualLogger(self.base_logger, new_context)
        


def create_contextual_logger(name: str, **context) -> ContextualLogger:
    """
    Create a contextual logger with the given context.
    
    Args:
        name: Logger name
        **context: Context key-value pairs
        
    Returns:
        ContextualLogger instance
    """
    base_logger = get_logger(name)
    return ContextualLogger(base_logger, context)

def setup_logging(config=None):
    """Setup logging configuration"""
    if config is None:
        config = {}
    
    # Handle both dictionary and direct level configurations
    if isinstance(config, dict):
        level_str = config.get('level', 'INFO')
        format_str = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Convert string level to logging constant
        if isinstance(level_str, str):
            level = getattr(logging, level_str.upper(), logging.INFO)
        else:
            level = level_str
    else:
        # Backward compatibility: if config is an integer, treat it as level
        level = config if isinstance(config, int) else logging.INFO
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str
    )
    