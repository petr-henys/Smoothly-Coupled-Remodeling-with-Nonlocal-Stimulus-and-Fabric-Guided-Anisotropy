"""
Tests for logger.py module (standard logging utilities).

Tests cover:
- setup_logging() with various configurations
- get_std_logger() name processing
- get_class_logger() for class instances
- Log level configuration
- File output
- Format strings
- Handler management
"""

import logging
import sys
import pytest
from simulation.logger import setup_logging, get_std_logger, get_class_logger


class TestSetupLogging:
    """Test logging setup and configuration."""
    
    def teardown_method(self):
        """Clean up logging handlers after each test."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)
    
    def test_default_setup(self):
        """Test default logging setup (WARNING level, console only)."""
        setup_logging()
    
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING
        assert len(root_logger.handlers) >= 1
        
        # Check console handler exists
        console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) >= 1
        assert console_handlers[0].stream == sys.stdout
    
    def test_debug_level(self):
        """Test DEBUG logging level."""
        setup_logging(level="DEBUG")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
    
    def test_warning_level(self):
        """Test WARNING logging level."""
        setup_logging(level="WARNING")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING
    
    def test_error_level(self):
        """Test ERROR logging level."""
        setup_logging(level="ERROR")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR
    
    def test_critical_level(self):
        """Test CRITICAL logging level."""
        setup_logging(level="CRITICAL")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.CRITICAL
    
    def test_case_insensitive_level(self):
        """Test that log level is case-insensitive."""
        setup_logging(level="debug")
        assert logging.getLogger().level == logging.DEBUG
        
        setup_logging(level="WaRnInG")
        assert logging.getLogger().level == logging.WARNING
    
    def test_invalid_level_raises_error(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level="INVALID_LEVEL")
    
    def test_file_output(self, tmp_path):
        """Test logging to file."""
        log_file = tmp_path / "test.log"
        setup_logging(level="INFO", log_file=str(log_file))
        
        # Log a message
        logger = logging.getLogger("test")
        logger.info("Test message")
        
        # Verify file was created and contains message
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Test message" in log_content
        assert "INFO" in log_content
    
    def test_file_output_creates_directory(self, tmp_path):
        """Test that file handler creates parent directories."""
        log_file = tmp_path / "subdir" / "nested" / "test.log"
        setup_logging(level="INFO", log_file=str(log_file))
        
        logger = logging.getLogger("test")
        logger.info("Test message")
        
        assert log_file.exists()
        assert log_file.parent.exists()
    
    def test_custom_format_string(self):
        """Test custom format string."""
        custom_format = "%(levelname)s - %(message)s"
        setup_logging(level="INFO", format_string=custom_format)
        
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        
        # Check formatter format string
        assert handler.formatter._fmt == custom_format
    
    def test_default_format_string(self):
        """Test default format string."""
        setup_logging()
        
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        
        expected_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert handler.formatter._fmt == expected_format
    
    def test_handlers_cleared_on_reconfigure(self):
        """Test that handlers are cleared when reconfiguring."""
        # First setup
        setup_logging(level="INFO")
        initial_count = len(logging.getLogger().handlers)
        
        # Second setup should clear and recreate
        setup_logging(level="DEBUG")
        new_count = len(logging.getLogger().handlers)
        
        # Should have same number of handlers (not duplicated)
        assert new_count == initial_count
    
    def test_console_and_file_handlers(self, tmp_path):
        """Test both console and file handlers are created."""
        log_file = tmp_path / "test.log"
        setup_logging(level="INFO", log_file=str(log_file))
        
        root_logger = logging.getLogger()
        
        # Should have both console and file handlers
        stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        
        assert len(stream_handlers) >= 1
        assert len(file_handlers) == 1
    
    def test_mpi_logger_suppressed(self):
        """Test that mpi4py.MPI logger is set to ERROR level."""
        setup_logging(level="DEBUG")
        
        mpi_logger = logging.getLogger('mpi4py.MPI')
        assert mpi_logger.level == logging.ERROR
    
    def test_matplotlib_logger_suppressed(self):
        """Test that matplotlib logger is set to WARNING level."""
        setup_logging(level="DEBUG")
        
        mpl_logger = logging.getLogger('matplotlib')
        assert mpl_logger.level == logging.WARNING
    
    def test_pil_logger_suppressed(self):
        """Test that PIL logger is set to WARNING level."""
        setup_logging(level="DEBUG")
        
        pil_logger = logging.getLogger('PIL')
        assert pil_logger.level == logging.WARNING


class TestGetLogger:
    """Test get_std_logger() function."""
    
    def test_get_logger_basic(self):
        """Test basic logger retrieval."""
        logger = get_std_logger("my_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "my_module"
    
    def test_get_logger_removes_src_prefix(self):
        """Test that 'src.' prefix is removed from logger names."""
        logger = get_std_logger("src.module.submodule")
        
        assert logger.name == "module.submodule"
    
    def test_get_logger_nested_module(self):
        """Test logger for nested module structure."""
        logger = get_std_logger("femurloader.process_gait_data")
        
        assert logger.name == "femurloader.process_gait_data"
    
    def test_get_logger_preserves_non_src_prefix(self):
        """Test that non-src prefixes are preserved."""
        logger = get_std_logger("tests.test_module")
        
        assert logger.name == "tests.test_module"
    
    def test_get_logger_returns_same_instance(self):
        """Test that get_logger returns the same instance for same name."""
        logger1 = get_std_logger("my_module")
        logger2 = get_std_logger("my_module")
        
        assert logger1 is logger2


class TestGetClassLogger:
    """Test get_class_logger() function."""
    
    def test_get_class_logger_basic(self):
        """Test basic class logger retrieval."""
        
        class MyClass:
            pass
        
        obj = MyClass()
        logger = get_class_logger(obj)
        
        assert isinstance(logger, logging.Logger)
        assert "MyClass" in logger.name
    
    def test_get_class_logger_with_module(self):
        """Test class logger includes module name."""
        
        class TestClass:
            pass
        
        obj = TestClass()
        logger = get_class_logger(obj)
        
        # Should include both module and class name
        assert "TestClass" in logger.name
    
    def test_get_class_logger_removes_src_prefix(self):
        """Test that 'src.' prefix is removed from class logger names."""
        
        class MyClass:
            __module__ = "src.my_module"
        
        obj = MyClass()
        logger = get_class_logger(obj)
        
        assert logger.name == "my_module.MyClass"
        assert not logger.name.startswith("src.")
    
    def test_get_class_logger_nested_module(self):
        """Test class logger for nested module structure."""
        
        class MyClass:
            __module__ = "femurloader.utils"
        
        obj = MyClass()
        logger = get_class_logger(obj)
        
        assert logger.name == "femurloader.utils.MyClass"


class TestLoggingFunctionality:
    """Test actual logging functionality."""
    
    def setup_method(self):
        """Set up logging for each test."""
        setup_logging(level="DEBUG")
    
    def teardown_method(self):
        """Clean up logging handlers."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
    
    def test_log_message_output(self, caplog):
        """Test that log messages are actually output."""
        logger = get_std_logger("test_module")
        
        with caplog.at_level(logging.INFO):
            logger.info("Test info message")
        
        assert "Test info message" in caplog.text
    
    def test_log_levels_respected(self, tmp_path):
        """Test that log levels are respected."""
        log_file = tmp_path / "test.log"
        setup_logging(level="WARNING", log_file=str(log_file))
        logger = get_std_logger("test_module")
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Check file output (caplog doesn't work with custom handlers)
        content = log_file.read_text()
        
        # DEBUG and INFO should not appear
        assert "Debug message" not in content
        assert "Info message" not in content
        
        # WARNING and ERROR should appear
        assert "Warning message" in content
        assert "Error message" in content
    
    def test_log_hierarchy(self, caplog):
        """Test that logger hierarchy works correctly."""
        parent_logger = get_std_logger("parent")
        child_logger = get_std_logger("parent.child")
        
        with caplog.at_level(logging.INFO):
            parent_logger.info("Parent message")
            child_logger.info("Child message")
        
        assert "Parent message" in caplog.text
        assert "Child message" in caplog.text


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def teardown_method(self):
        """Clean up logging handlers."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
    
    def test_empty_logger_name(self):
        """Test get_logger with empty name returns root logger."""
        logger = get_std_logger("")
        
        assert isinstance(logger, logging.Logger)
        # Empty name returns root logger
        assert logger.name == "root"
    
    def test_none_format_string(self):
        """Test that None format string uses default."""
        setup_logging(format_string=None)
        
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        
        # Should use default format
        assert "%(asctime)s" in handler.formatter._fmt
    
    def test_multiple_setup_calls(self, tmp_path):
        """Test multiple setup_logging calls don't accumulate handlers."""
        log_file = tmp_path / "test.log"
        
        for _ in range(3):
            setup_logging(level="INFO", log_file=str(log_file))
        
        root_logger = logging.getLogger()
        
        # Should have exactly 2 handlers (console + file), not 6
        assert len(root_logger.handlers) == 2
    
    def test_file_handler_with_existing_file(self, tmp_path):
        """Test file handler appends to existing file."""
        log_file = tmp_path / "test.log"
        
        # First setup and log
        setup_logging(level="INFO", log_file=str(log_file))
        logger = logging.getLogger("test")
        logger.info("First message")
        
        # Second setup and log
        logging.getLogger().handlers.clear()
        setup_logging(level="INFO", log_file=str(log_file))
        logger.info("Second message")
        
        # Both messages should be in file
        content = log_file.read_text()
        assert "First message" in content
        assert "Second message" in content


class TestIntegration:
    """Integration tests for logging system."""
    
    def teardown_method(self):
        """Clean up logging handlers."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
    
    def test_complete_logging_workflow(self, tmp_path):
        """Test complete logging workflow from setup to output."""
        log_file = tmp_path / "workflow.log"
        
        # Setup logging
        setup_logging(level="DEBUG", log_file=str(log_file))
        
        # Get different types of loggers
        module_logger = get_std_logger("my_module")
        
        class TestClass:
            pass
        
        obj = TestClass()
        class_logger = get_class_logger(obj)
        
        # Log messages at different levels
        module_logger.debug("Module debug")
        module_logger.info("Module info")
        class_logger.warning("Class warning")
        class_logger.error("Class error")
        
        # Verify file output
        file_content = log_file.read_text()
        assert "Module debug" in file_content
        assert "Module info" in file_content
        assert "Class warning" in file_content
        assert "Class error" in file_content
        
        # Verify log levels in file
        assert "DEBUG" in file_content
        assert "INFO" in file_content
        assert "WARNING" in file_content
        assert "ERROR" in file_content
    
    def test_logger_name_consistency(self):
        """Test that logger names are consistent across different retrieval methods."""
        setup_logging()
        
        # Get logger multiple times
        logger1 = get_std_logger("test.module")
        logger2 = get_std_logger("test.module")
        logger3 = logging.getLogger("test.module")
        
        # All should be the same instance
        assert logger1 is logger2
        assert logger2 is logger3
