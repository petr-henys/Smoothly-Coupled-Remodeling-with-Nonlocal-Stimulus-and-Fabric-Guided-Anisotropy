"""
Centralized logging configuration for the FemurLoader project.

This module provides a unified logging setup to ensure consistent logging
behavior across all modules in the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "WARNING",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the entire project.
    
    Parameters
    ----------
    level : str, default "WARNING"
        Logging level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    log_file : str, optional
        Path to log file. If None, logs only to console.
    format_string : str, optional
        Custom format string. If None, uses default format.
    """
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
    """
    Get a logger with the specified name.
    
    Parameters
    ----------
    name : str
        Logger name, typically __name__ for module loggers
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
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
    """
    Get a logger for a class instance.
    
    Parameters
    ----------
    cls : object
        Class instance
        
    Returns
    -------
    logging.Logger
        Configured logger instance with class name
    """
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


# Default initialization with WARNING level
if not logging.getLogger().handlers:
    setup_logging(level="WARNING")
