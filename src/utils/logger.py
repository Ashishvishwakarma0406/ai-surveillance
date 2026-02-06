"""
Logger Module

Provides colored console logging and file logging for the AI Surveillance System.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    # Color mapping for log levels
    COLORS = {
        'DEBUG': Fore.CYAN if COLORAMA_AVAILABLE else '',
        'INFO': Fore.GREEN if COLORAMA_AVAILABLE else '',
        'WARNING': Fore.YELLOW if COLORAMA_AVAILABLE else '',
        'ERROR': Fore.RED if COLORAMA_AVAILABLE else '',
        'CRITICAL': Fore.RED + Style.BRIGHT if COLORAMA_AVAILABLE else '',
    }
    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ''
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for level
        color = self.COLORS.get(record.levelname, '')
        
        # Format timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Build formatted message
        level = f"{color}[{record.levelname}]{self.RESET}"
        message = f"{timestamp} {level} {record.getMessage()}"
        
        return message


class AlertFormatter(logging.Formatter):
    """Special formatter for alert messages."""
    
    SEVERITY_COLORS = {
        'CRITICAL': Fore.RED + Style.BRIGHT if COLORAMA_AVAILABLE else '',
        'WARNING': Fore.YELLOW if COLORAMA_AVAILABLE else '',
        'INFORMATIONAL': Fore.BLUE if COLORAMA_AVAILABLE else '',
    }
    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ''
    
    def format(self, record: logging.LogRecord) -> str:
        """Format alert message with appropriate styling."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Check if this is an alert record with severity
        severity = getattr(record, 'severity', 'INFO')
        color = self.SEVERITY_COLORS.get(severity.upper(), '')
        
        # Build alert banner
        if severity.upper() == 'CRITICAL':
            banner = f"{color}{'!'*60}{self.RESET}"
            message = f"\n{banner}\n{color}⚠️  ALERT: {record.getMessage()}{self.RESET}\n{banner}"
        else:
            message = f"{color}[{severity}] {timestamp} - {record.getMessage()}{self.RESET}"
        
        return message


class SurveillanceLogger:
    """
    Logger manager for the AI Surveillance System.
    
    Provides separate loggers for system events and alerts.
    """
    
    _instance: Optional['SurveillanceLogger'] = None
    
    def __new__(cls) -> 'SurveillanceLogger':
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize loggers."""
        if self._initialized:
            return
        
        self._initialized = True
        self.system_logger = logging.getLogger('surveillance.system')
        self.alert_logger = logging.getLogger('surveillance.alert')
        
        # Prevent duplicate handlers
        self.system_logger.handlers = []
        self.alert_logger.handlers = []
        
        # Set default levels
        self.system_logger.setLevel(logging.DEBUG)
        self.alert_logger.setLevel(logging.INFO)
    
    def setup(
        self,
        log_dir: str = "logs",
        console_output: bool = True,
        file_output: bool = True,
        debug_mode: bool = False
    ) -> None:
        """
        Setup logging handlers.
        
        Args:
            log_dir: Directory for log files
            console_output: Enable console logging
            file_output: Enable file logging
            debug_mode: Enable debug level logging
        """
        # Ensure log directory exists
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Set log level
        level = logging.DEBUG if debug_mode else logging.INFO
        self.system_logger.setLevel(level)
        
        # Console handler for system logger
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColoredFormatter())
            console_handler.setLevel(level)
            self.system_logger.addHandler(console_handler)
            
            # Console handler for alert logger
            alert_console = logging.StreamHandler(sys.stdout)
            alert_console.setFormatter(AlertFormatter())
            self.alert_logger.addHandler(alert_console)
        
        # File handler for system logger
        if file_output:
            system_file = log_path / "system.log"
            file_handler = logging.FileHandler(system_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            self.system_logger.addHandler(file_handler)
            
            # File handler for alert logger
            alert_file = log_path / "alerts.log"
            alert_file_handler = logging.FileHandler(alert_file, encoding='utf-8')
            alert_file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            self.alert_logger.addHandler(alert_file_handler)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.system_logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.system_logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.system_logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.system_logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.system_logger.critical(message)
    
    def alert(self, message: str, severity: str = "INFO") -> None:
        """
        Log an alert message.
        
        Args:
            message: Alert message
            severity: Alert severity (CRITICAL, WARNING, INFORMATIONAL)
        """
        record = self.alert_logger.makeRecord(
            'surveillance.alert',
            logging.INFO,
            '', 0, message, (), None
        )
        record.severity = severity
        self.alert_logger.handle(record)


# Global logger instance
_logger = SurveillanceLogger()


def setup_logger(
    log_dir: str = "logs",
    console_output: bool = True,
    file_output: bool = True,
    debug_mode: bool = False
) -> SurveillanceLogger:
    """
    Setup and configure the logger.
    
    Args:
        log_dir: Directory for log files
        console_output: Enable console logging
        file_output: Enable file logging
        debug_mode: Enable debug level logging
        
    Returns:
        Configured logger instance
    """
    _logger.setup(log_dir, console_output, file_output, debug_mode)
    return _logger


def get_logger() -> SurveillanceLogger:
    """
    Get the global logger instance.
    
    Returns:
        Logger instance
    """
    return _logger
