# -*- coding=utf-8 -*-
"""
Atomic Writer Configuration

This module provides configuration management for the AtomicJSONWriter,
including validation, retry logic, and file system checks.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Literal, Callable, Any, List
from pathlib import Path
import json

try:
    from .error_handling import AtomicWriteErrorHandler, ErrorContext, TempFileCleanupManager, FallbackWriteHandler
except ImportError:
    from error_handling import AtomicWriteErrorHandler, ErrorContext, TempFileCleanupManager, FallbackWriteHandler


@dataclass
class AtomicWriterConfig:
    """Configuration for AtomicJSONWriter operations."""

    temp_dir: Optional[str] = None
    sync_mode: Literal["fsync", "fdatasync", "none"] = "fsync"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 0.1
    cleanup_temp_files: bool = True
    min_disk_space_mb: int = 1
    validate_permissions: bool = True
    use_file_locking: bool = True
    lock_timeout: float = 30.0

    def __post_init__(self):
        """Validate configuration values after initialization."""
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if self.min_disk_space_mb < 0:
            raise ValueError("min_disk_space_mb must be non-negative")
        if self.sync_mode not in ("fsync", "fdatasync", "none"):
            raise ValueError("sync_mode must be 'fsync', 'fdatasync', or 'none'")


class FileSystemValidator:
    """Validates file system conditions for atomic writing operations."""

    def __init__(self, config: AtomicWriterConfig):
        """
        Initialize the validator with configuration.
        
        Args:
            config: Configuration for validation behavior
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_write_conditions(self, filepath: Path) -> None:
        """
        Validate that all conditions are met for writing to the specified file.
        
        Args:
            filepath: Target file path
            
        Raises:
            PermissionError: If write permissions are insufficient
            OSError: If disk space is insufficient or other filesystem issues
        """
        if self.config.validate_permissions:
            self._validate_permissions(filepath)

        self._validate_disk_space(filepath)

    def _validate_permissions(self, filepath: Path) -> None:
        """
        Validate write permissions for the target file and directory.
        
        Args:
            filepath: Target file path
            
        Raises:
            PermissionError: If write permissions are insufficient
        """
        # Check parent directory
        parent_dir = filepath.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise PermissionError(f"Cannot create directory {parent_dir}: {e}")

        if not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory {parent_dir}")

        # Check target file if it exists
        if filepath.exists() and not os.access(filepath, os.W_OK):
            raise PermissionError(f"No write permission for file {filepath}")

        # Check temp directory if specified
        if self.config.temp_dir:
            temp_dir = Path(self.config.temp_dir)
            if not temp_dir.exists():
                try:
                    temp_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise PermissionError(f"Cannot create temp directory {temp_dir}: {e}")

            if not os.access(temp_dir, os.W_OK):
                raise PermissionError(f"No write permission for temp directory {temp_dir}")

    def _validate_disk_space(self, filepath: Path) -> None:
        """
        Validate that sufficient disk space is available.
        
        Args:
            filepath: Target file path
            
        Raises:
            OSError: If insufficient disk space is available
        """
        if self.config.min_disk_space_mb <= 0:
            return

        try:
            # Check space in target directory
            parent_dir = filepath.parent
            statvfs = os.statvfs(parent_dir)
            available_bytes = statvfs.f_frsize * statvfs.f_bavail
            required_bytes = self.config.min_disk_space_mb * 1024 * 1024

            if available_bytes < required_bytes:
                raise OSError(
                    f"Insufficient disk space: {available_bytes / (1024*1024):.1f}MB available, "
                    f"{self.config.min_disk_space_mb}MB required"
                )

            # Also check temp directory if different
            if self.config.temp_dir:
                temp_dir = Path(self.config.temp_dir)
                if temp_dir.resolve() != parent_dir.resolve():
                    statvfs = os.statvfs(temp_dir)
                    available_bytes = statvfs.f_frsize * statvfs.f_bavail

                    if available_bytes < required_bytes:
                        raise OSError(
                            f"Insufficient disk space in temp directory: "
                            f"{available_bytes / (1024*1024):.1f}MB available, "
                            f"{self.config.min_disk_space_mb}MB required"
                        )

        except (OSError, AttributeError) as e:
            if isinstance(e, OSError) and "Insufficient disk space" in str(e):
                raise
            # statvfs not available on all platforms, log warning but continue
            self.logger.warning(f"Could not check disk space: {e}")


class RetryHandler:
    """Handles retry logic for atomic write operations."""

    def __init__(self, config: AtomicWriterConfig):
        """
        Initialize the retry handler with configuration.
        
        Args:
            config: Configuration for retry behavior
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def execute_with_retry(self, operation: Callable[[], Any], operation_name: str = "operation") -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Function to execute
            operation_name: Name of the operation for logging
            
        Returns:
            Any: Result of the operation
            
        Raises:
            Exception: The last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return operation()
            except Exception as e:
                last_exception = e

                if attempt < self.config.max_retries:
                    # Determine if this is a retryable error
                    if self._is_retryable_error(e):
                        delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(
                            f"{operation_name} failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        # Non-retryable error, fail immediately
                        self.logger.error(f"{operation_name} failed with non-retryable error: {e}")
                        raise
                else:
                    # All retries exhausted
                    self.logger.error(f"{operation_name} failed after {self.config.max_retries + 1} attempts: {e}")
                    raise

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Args:
            error: The exception to check
            
        Returns:
            bool: True if the error is retryable
        """
        # Retryable errors are typically transient filesystem issues
        retryable_errors = (
            OSError,  # General OS errors (might be transient)
            IOError,  # I/O errors (might be transient)
        )

        # Non-retryable errors
        non_retryable_messages = [
            "permission denied",
            "no such file or directory",
            "not a directory",
            "file exists",
            "invalid argument",
        ]

        if isinstance(error, retryable_errors):
            error_msg = str(error).lower()
            # Check if it's a non-retryable OS error
            for msg in non_retryable_messages:
                if msg in error_msg:
                    return False
            return True

        # Permission errors and value errors are not retryable
        if isinstance(error, (PermissionError, ValueError, TypeError)):
            return False

        return False


class ConfiguredAtomicWriter:
    """
    AtomicJSONWriter with configuration support and enhanced error handling.
    
    This class wraps the basic AtomicJSONWriter with configuration management,
    validation, and retry logic.
    """

    def __init__(self, config: Optional[AtomicWriterConfig] = None):
        """
        Initialize the configured atomic writer.
        
        Args:
            config: Configuration for the writer. If None, uses default config.
        """
        self.config = config or AtomicWriterConfig()
        self.validator = FileSystemValidator(self.config)
        self.retry_handler = RetryHandler(self.config)
        self.logger = logging.getLogger(__name__)

        # Initialize comprehensive error handling
        self.error_handler = AtomicWriteErrorHandler(__name__)
        self.cleanup_manager = TempFileCleanupManager(__name__)
        self.fallback_handler = FallbackWriteHandler(__name__)

        # Import here to avoid circular imports
        try:
            from .atomic_json_writer import AtomicJSONWriter
        except ImportError:
            from atomic_json_writer import AtomicJSONWriter
        self._writer = AtomicJSONWriter(temp_dir=self.config.temp_dir)

    def write_atomic(self, data: dict, filepath: str | Path,
            mode="w+", cls=json.JSONEncoder) -> bool:
        """
        Write JSON data atomically with comprehensive error handling and fallback mechanisms.
        
        Args:
            data: Dictionary to write as JSON
            filepath: Target file path
            
        Returns:
            bool: True if write was successful
            
        Raises:
            Exception: If write fails after all retries and fallbacks
        """
        filepath_obj = Path(filepath).resolve()

        # Clean up any orphaned temp files first
        if self.config.cleanup_temp_files:
            self.cleanup_manager.cleanup_orphaned_temp_files(
                filepath_obj.parent,
                max_age_hours=1  # Clean up temp files older than 1 hour
            )

        try:
            # Pre-validate conditions
            self.validator.validate_write_conditions(filepath_obj)

            # Execute write with retry logic
            def write_operation():
                return self._writer.write_atomic(
                    data,
                    str(filepath_obj),
                    self.config.sync_mode,
                    self.config.use_file_locking,
                    mode=mode,
                    cls=cls
                )

            return self.retry_handler.execute_with_retry(
                write_operation,
                f"atomic write to {filepath_obj}"
            )

        except Exception as e:
            # Create error context
            context = ErrorContext(
                operation="atomic_write",
                target_file=str(filepath_obj),
                error_type=type(e).__name__,
                error_message=str(e)
            )

            # Define fallback actions
            def fallback_direct():
                return self.fallback_handler.fallback_direct_write(data, filepath_obj)

            def fallback_backup():
                return self.fallback_handler.fallback_backup_write(data, filepath_obj)

            # Try fallback mechanisms
            if self.error_handler.handle_error(e, context, fallback_direct):
                self.logger.info(f"Atomic write succeeded via direct fallback for {filepath_obj}")
                return True

            # If direct fallback failed, try backup fallback
            context.retry_count = 1
            if self.error_handler.handle_error(e, context, fallback_backup):
                self.logger.warning(f"Atomic write succeeded via backup fallback for {filepath_obj}")
                return True

            # All fallbacks failed, re-raise the original exception
            self.logger.error(f"All write attempts and fallbacks failed for {filepath_obj}")
            raise

    def get_error_summary(self, hours: int = 24) -> dict:
        """
        Get summary of recent errors.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            dict: Error summary
        """
        return self.error_handler.get_error_summary(hours)

    def cleanup_temp_files(self, directory: Optional[Path] = None, max_age_hours: int = 24) -> List[str]:
        """
        Manually clean up orphaned temporary files.
        
        Args:
            directory: Directory to clean up. If None, uses temp_dir from config
            max_age_hours: Maximum age of temp files to keep
            
        Returns:
            List[str]: List of cleaned up files
        """
        if directory is None:
            if self.config.temp_dir:
                directory = Path(self.config.temp_dir)
            else:
                self.logger.warning("No directory specified for temp file cleanup")
                return []

        return self.cleanup_manager.cleanup_orphaned_temp_files(directory, max_age_hours)

    def validate_file_integrity(self, filepath: Path) -> bool:
        """
        Validate the integrity of a JSON file.
        
        Args:
            filepath: Path to file to validate
            
        Returns:
            bool: True if file is valid
        """
        return self.cleanup_manager.validate_json_integrity(filepath)

    def attempt_file_recovery(self, filepath: Path) -> bool:
        """
        Attempt to recover a corrupted file from backup.
        
        Args:
            filepath: Path to corrupted file
            
        Returns:
            bool: True if recovery succeeded
        """
        return self.cleanup_manager.attempt_recovery(filepath)