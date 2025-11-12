"""DEPRECATED: Standard Python logging setup for non-MPI modules.

This module is deprecated. Use simulation.logger for all new code.
The logger.py module provides MPI-safe logging with rank-0 only output
and lazy string evaluation, which is the recommended approach for this codebase.

This module is retained only for backward compatibility with existing tests.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """Configure root logger with console and optional file output."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplication
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress MPI warnings
    logging.getLogger('mpi4py.MPI').setLevel(logging.ERROR)
    
    # Set specific library loggers to appropriate levels
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger with cleaned module name."""
    # Clean up module names for better readability
    if name == "__main__":
        # For main scripts, use the script filename
        import __main__
        if hasattr(__main__, '__file__') and __main__.__file__:
            script_name = Path(__main__.__file__).stem
            logger_name = f"main.{script_name}"
        else:
            logger_name = "main"
    elif name.startswith("src."):
        # Remove 'src.' prefix for cleaner names
        logger_name = name[4:]
    else:
        logger_name = name
    
    return logging.getLogger(logger_name)


def get_class_logger(cls) -> logging.Logger:
    """Get logger for class instance with module.ClassName format."""
    module_name = cls.__class__.__module__
    class_name = cls.__class__.__name__
    
    # If module is __main__, use just the class name to avoid showing __main__
    if module_name == "__main__":
        logger_name = class_name
    else:
        # Remove 'src.' prefix if present for cleaner names
        if module_name.startswith("src."):
            module_name = module_name[4:]
        logger_name = f"{module_name}.{class_name}"
    
    return logging.getLogger(logger_name)


# Default initialization with INFO level
if not logging.getLogger().handlers:
    setup_logging()
