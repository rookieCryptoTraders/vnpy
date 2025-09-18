# -*- coding=utf-8 -*-
"""
Comprehensive Error Handling and Logging

This module provides enhanced error handling, logging, and recovery utilities
for atomic JSON writing operations. It includes fallback mechanisms and
cleanup utilities for orphaned temporary files.
"""

import os
import glob
import time
import logging
import traceback
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for categorizing different types of errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error reporting and recovery."""

    operation: str
    target_file: str
    temp_file: Optional[str] = None
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    timestamp: float = 0.0
    stack_trace: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class AtomicWriteErrorHandler:
    """
    Comprehensive error handler for atomic write operations.

    Provides detailed error logging, categorization, and recovery mechanisms
    for various failure scenarios during atomic JSON writing.
    """

    def __init__(self, logger_name: str = __name__):
        """
        Initialize the error handler.

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)
        self.error_history: List[ErrorContext] = []
        self.max_error_history = 100

        # Configure detailed logging format
        self._setup_detailed_logging()

    def _setup_detailed_logging(self):
        """Set up detailed logging format for error reporting."""
        # Only set up if no handlers exist to avoid duplicate logs
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        fallback_action: Optional[Callable] = None,
    ) -> bool:
        """
        Handle an error with comprehensive logging and optional fallback.

        Args:
            error: The exception that occurred
            context: Context information about the error
            fallback_action: Optional fallback function to execute

        Returns:
            bool: True if error was handled successfully (including fallback)
        """
        # Update context with error details
        context.error_type = type(error).__name__
        context.error_message = str(error)
        context.stack_trace = traceback.format_exc()
        context.severity = self._categorize_error(error)

        # Add to error history
        self._add_to_history(context)

        # Log the error with appropriate level
        self._log_error(error, context)

        # Attempt fallback if provided
        if fallback_action:
            try:
                self.logger.info(f"Attempting fallback action for {context.operation}")
                fallback_action()
                self.logger.info(f"Fallback action succeeded for {context.operation}")
                return True
            except Exception as fallback_error:
                self.logger.error(
                    f"Fallback action failed for {context.operation}: {fallback_error}"
                )

        return False

    def _categorize_error(self, error: Exception) -> ErrorSeverity:
        """
        Categorize error severity based on error type and message.

        Args:
            error: The exception to categorize

        Returns:
            ErrorSeverity: Severity level of the error
        """
        error_msg = str(error).lower()

        # Critical errors that indicate serious system issues
        if isinstance(error, (SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL

        if any(
            keyword in error_msg
            for keyword in ["no space left", "disk full", "filesystem full"]
        ):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if isinstance(error, (PermissionError, OSError)):
            if any(
                keyword in error_msg
                for keyword in ["permission denied", "access denied", "read-only"]
            ):
                return ErrorSeverity.HIGH

        # Medium severity errors (default for most filesystem issues)
        if isinstance(error, (OSError, IOError)):
            return ErrorSeverity.MEDIUM

        # Low severity errors (typically configuration or data issues)
        if isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM

    def _log_error(self, error: Exception, context: ErrorContext):
        """
        Log error with appropriate level and detailed information.

        Args:
            error: The exception that occurred
            context: Context information about the error
        """
        base_msg = (
            f"Atomic write error in {context.operation}: {context.error_message} "
            f"(target: {context.target_file})"
        )

        if context.temp_file:
            base_msg += f" (temp: {context.temp_file})"

        if context.retry_count > 0:
            base_msg += f" (retry {context.retry_count})"

        # Log with appropriate level based on severity
        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(base_msg)
            self.logger.critical(f"Stack trace:\n{context.stack_trace}")
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(base_msg)
            self.logger.debug(f"Stack trace:\n{context.stack_trace}")
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(base_msg)
            self.logger.debug(f"Stack trace:\n{context.stack_trace}")
        else:  # LOW
            self.logger.info(base_msg)
            self.logger.debug(f"Stack trace:\n{context.stack_trace}")

    def _add_to_history(self, context: ErrorContext):
        """
        Add error context to history, maintaining size limit.

        Args:
            context: Error context to add
        """
        self.error_history.append(context)

        # Maintain history size limit
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history :]

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of errors from the specified time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Dict: Error summary statistics
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_errors = [
            error for error in self.error_history if error.timestamp >= cutoff_time
        ]

        if not recent_errors:
            return {"total_errors": 0, "period_hours": hours}

        # Count by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = sum(
                1 for error in recent_errors if error.severity == severity
            )

        # Count by error type
        error_type_counts = {}
        for error in recent_errors:
            error_type_counts[error.error_type] = (
                error_type_counts.get(error.error_type, 0) + 1
            )

        # Most common operations with errors
        operation_counts = {}
        for error in recent_errors:
            operation_counts[error.operation] = (
                operation_counts.get(error.operation, 0) + 1
            )

        return {
            "total_errors": len(recent_errors),
            "period_hours": hours,
            "severity_breakdown": severity_counts,
            "error_types": error_type_counts,
            "operations": operation_counts,
            "first_error": recent_errors[0].timestamp if recent_errors else None,
            "last_error": recent_errors[-1].timestamp if recent_errors else None,
        }


class TempFileCleanupManager:
    """
    Manager for cleaning up orphaned temporary files from failed atomic writes.
    """

    def __init__(self, logger_name: str = __name__):
        """
        Initialize the cleanup manager.

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)

    def cleanup_orphaned_temp_files(
        self, directory: Path, max_age_hours: int = 24, dry_run: bool = False
    ) -> List[str]:
        """
        Clean up orphaned temporary files in the specified directory.

        Args:
            directory: Directory to clean up
            max_age_hours: Maximum age of temp files to keep (in hours)
            dry_run: If True, only report what would be cleaned up

        Returns:
            List[str]: List of files that were (or would be) cleaned up
        """
        if not directory.exists():
            self.logger.debug(f"Directory {directory} does not exist, skipping cleanup")
            return []

        # Find temporary files (files with .tmp. in the name)
        temp_patterns = ["*.tmp.*", ".*.tmp.*", "*.backup"]

        temp_files = []
        for pattern in temp_patterns:
            temp_files.extend(directory.glob(pattern))

        if not temp_files:
            self.logger.debug(f"No temporary files found in {directory}")
            return []

        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_files = []

        for temp_file in temp_files:
            try:
                # Check file age
                file_mtime = temp_file.stat().st_mtime

                if file_mtime < cutoff_time:
                    if dry_run:
                        self.logger.info(
                            f"Would clean up orphaned temp file: {temp_file}"
                        )
                        cleaned_files.append(str(temp_file))
                    else:
                        temp_file.unlink()
                        self.logger.info(f"Cleaned up orphaned temp file: {temp_file}")
                        cleaned_files.append(str(temp_file))
                else:
                    self.logger.debug(f"Temp file {temp_file} is too recent, keeping")

            except OSError as e:
                self.logger.warning(f"Failed to process temp file {temp_file}: {e}")

        if cleaned_files:
            self.logger.info(
                f"Cleaned up {len(cleaned_files)} orphaned temp files in {directory}"
            )

        return cleaned_files

    def validate_json_integrity(self, filepath: Path) -> bool:
        """
        Validate that a JSON file is properly formatted and complete.

        Args:
            filepath: Path to JSON file to validate

        Returns:
            bool: True if file is valid JSON
        """
        if not filepath.exists():
            self.logger.debug(f"File {filepath} does not exist")
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                import json

                json.load(f)

            self.logger.debug(f"JSON file {filepath} is valid")
            return True

        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            self.logger.warning(f"JSON file {filepath} is invalid: {e}")
            return False

    def attempt_recovery(self, filepath: Path, backup_suffix: str = ".backup") -> bool:
        """
        Attempt to recover a corrupted file from backup.

        Args:
            filepath: Path to the corrupted file
            backup_suffix: Suffix for backup files

        Returns:
            bool: True if recovery was successful
        """
        backup_path = filepath.with_suffix(filepath.suffix + backup_suffix)

        if not backup_path.exists():
            self.logger.warning(f"No backup file found for {filepath}")
            return False

        # Validate backup file
        if not self.validate_json_integrity(backup_path):
            self.logger.error(f"Backup file {backup_path} is also corrupted")
            return False

        try:
            # Copy backup to original location
            import shutil

            shutil.copy2(backup_path, filepath)

            # Validate recovered file
            if self.validate_json_integrity(filepath):
                self.logger.info(f"Successfully recovered {filepath} from backup")
                return True
            else:
                self.logger.error(
                    f"Recovery failed: restored file {filepath} is invalid"
                )
                return False

        except OSError as e:
            self.logger.error(f"Failed to recover {filepath} from backup: {e}")
            return False


class FallbackWriteHandler:
    """
    Provides fallback mechanisms when atomic writing fails.
    """

    def __init__(self, logger_name: str = __name__):
        """
        Initialize the fallback handler.

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)

    def fallback_direct_write(self, data: Dict[str, Any], filepath: Path) -> bool:
        """
        Fallback to direct file writing when atomic write fails.

        Args:
            data: Data to write
            filepath: Target file path

        Returns:
            bool: True if fallback write succeeded
        """
        self.logger.warning(f"Attempting fallback direct write to {filepath}")

        try:
            import json

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
                f.flush()

            self.logger.info(f"Fallback direct write succeeded for {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Fallback direct write failed for {filepath}: {e}")
            return False

    def fallback_backup_write(self, data: Dict[str, Any], filepath: Path) -> bool:
        """
        Fallback to writing to a backup location when primary write fails.

        Args:
            data: Data to write
            filepath: Original target file path

        Returns:
            bool: True if backup write succeeded
        """
        backup_path = filepath.with_suffix(filepath.suffix + ".fallback")
        self.logger.warning(f"Attempting fallback backup write to {backup_path}")

        try:
            import json

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
                f.flush()

            self.logger.info(f"Fallback backup write succeeded: {backup_path}")
            return True

        except Exception as e:
            self.logger.error(f"Fallback backup write failed for {backup_path}: {e}")
            return False
