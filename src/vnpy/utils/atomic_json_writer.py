# -*- coding=utf-8 -*-
"""
Atomic JSON Writer Utility

This module provides atomic JSON file writing capabilities to ensure data integrity
even during unexpected program termination. It uses temporary files and atomic
rename operations to guarantee that JSON files are never left in a corrupted state.
"""

import json
import os
import tempfile
import shutil
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from .file_locking import acquire_file_lock
except ImportError:
    from file_locking import acquire_file_lock


class AtomicJSONWriterError(Exception):
    """Base exception for AtomicJSONWriter errors"""
    pass


class DiskSpaceError(AtomicJSONWriterError):
    """Raised when insufficient disk space is available"""
    pass


class PermissionError(AtomicJSONWriterError):
    """Raised when file permissions prevent writing"""
    pass


class AtomicJSONWriter:
    """
    Atomic JSON file writer that ensures data integrity during write operations.
    
    Uses temporary files and atomic rename operations to guarantee that JSON files
    are never left in a corrupted state, even if the program is interrupted during
    writing.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the AtomicJSONWriter.
        
        Args:
            temp_dir: Optional directory for temporary files. If None, uses the
                     same directory as the target file.
        """
        self.temp_dir = temp_dir
        self.logger = logging.getLogger(__name__)
        
    def write_atomic(self, data: Dict[str, Any], filepath: str, sync_mode: str = "fsync", use_locking: bool = True,
            mode="w+", cls=json.JSONEncoder) -> bool:
        """
        Atomically write JSON data to a file with optional file locking.
        
        This method writes data to a temporary file first, then atomically
        renames it to the target file. This ensures that the target file
        is never left in a partially written state.
        
        Args:
            data: Dictionary to write as JSON
            filepath: Target file path
            sync_mode: Sync mode - "fsync", "fdatasync", or "none"
            use_locking: Whether to use file locking for concurrent access protection
            cls: Optional JSON encoder class
            mode: File open mode

        Returns:
            bool: True if write was successful, False otherwise
            
        Raises:
            AtomicJSONWriterError: If the write operation fails
        """
        filepath = Path(filepath).resolve()
        
        try:
            if use_locking:
                # Use file locking to prevent concurrent writes
                with acquire_file_lock(str(filepath), timeout=30.0):
                    return self._write_atomic_internal(filepath, data, sync_mode, mode=mode, cls=cls)
            else:
                # Write without locking
                return self._write_atomic_internal(filepath, data, sync_mode, mode=mode, cls=cls)
                
        except Exception as e:
            self.logger.error(f"Failed to write JSON data to {filepath}: {e}")
            raise AtomicJSONWriterError(f"Atomic write failed: {e}") from e
    
    def _write_atomic_internal(self, filepath: Path, data: Dict[str, Any], sync_mode: str,
            mode="w+", cls=json.JSONEncoder) -> bool:
        """
        Internal method for atomic writing without locking.
        
        Args:
            filepath: Target file path
            data: Dictionary to write as JSON
            sync_mode: Sync mode - "fsync", "fdatasync", or "none"
            cls: Optional JSON encoder class
            mode: File open mode

        Returns:
            bool: True if write was successful
        """
        # Validate preconditions
        self._validate_write_conditions(filepath)
        
        # Create temporary file
        temp_path = self._create_temp_file(filepath)
        
        try:
            # Write data and sync to disk
            self._write_and_sync(temp_path, data, sync_mode, mode=mode, cls=cls)
            
            # Atomically replace target with temp file
            self._atomic_replace(temp_path, filepath)
            
            self.logger.debug(f"Successfully wrote JSON data to {filepath}")
            return True
            
        except Exception as e:
            # Clean up temp file on failure
            self._cleanup_temp_file(temp_path)
            raise e
    
    def _validate_write_conditions(self, filepath: Path) -> None:
        """
        Validate that write conditions are met.
        
        Args:
            filepath: Target file path
            
        Raises:
            PermissionError: If write permissions are insufficient
            DiskSpaceError: If insufficient disk space is available
        """
        # Check parent directory exists and is writable
        parent_dir = filepath.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise PermissionError(f"Cannot create directory {parent_dir}: {e}")
        
        if not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory {parent_dir}")
        
        # Check if target file exists and is writable
        if filepath.exists() and not os.access(filepath, os.W_OK):
            raise PermissionError(f"No write permission for file {filepath}")
        
        # Check available disk space (basic check)
        try:
            statvfs = os.statvfs(parent_dir)
            available_bytes = statvfs.f_frsize * statvfs.f_bavail
            # Require at least 1MB free space as a safety margin
            if available_bytes < 1024 * 1024:
                raise DiskSpaceError(f"Insufficient disk space: {available_bytes} bytes available")
        except (OSError, AttributeError):
            # statvfs not available on all platforms, skip check
            pass
    
    def _create_temp_file(self, target_path: Path) -> Path:
        """
        Create a temporary file in the same directory as the target.
        
        Args:
            target_path: Path to the target file
            
        Returns:
            Path: Path to the created temporary file
            
        Raises:
            AtomicJSONWriterError: If temp file creation fails
        """
        try:
            # Use target directory or specified temp directory
            temp_dir = self.temp_dir if self.temp_dir else target_path.parent
            
            # Create temporary file with appropriate prefix
            prefix = f".{target_path.name}.tmp."
            fd, temp_path = tempfile.mkstemp(
                prefix=prefix,
                dir=temp_dir,
                text=True
            )
            
            # Close the file descriptor since we'll open it again for writing
            os.close(fd)
            
            temp_path = Path(temp_path)
            self.logger.debug(f"Created temporary file: {temp_path}")
            return temp_path
            
        except OSError as e:
            raise AtomicJSONWriterError(f"Failed to create temporary file: {e}")
    
    def _write_and_sync(self, temp_path: Path, data: Dict[str, Any], sync_mode: str = "fsync",
            mode="w+", cls=json.JSONEncoder) -> None:
        """
        Write JSON data to temporary file and sync to disk.
        
        Args:
            temp_path: Path to temporary file
            data: Data to write as JSON
            sync_mode: Sync mode - "fsync", "fdatasync", or "none"
            cls: Optional JSON encoder class
            mode: File open mode
            
        Raises:
            AtomicJSONWriterError: If write or sync fails
        """
        try:
            with open(temp_path, mode, encoding='utf-8') as f:
                # Write JSON data with proper formatting
                json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=False, cls=cls)
                
                # Force write to disk based on sync mode
                f.flush()
                if sync_mode == "fsync":
                    os.fsync(f.fileno())
                elif sync_mode == "fdatasync":
                    # Use fdatasync if available, fallback to fsync
                    if hasattr(os, 'fdatasync'):
                        os.fdatasync(f.fileno())
                    else:
                        os.fsync(f.fileno())
                # sync_mode == "none" means no explicit sync
                
            self.logger.debug(f"Successfully wrote and synced data to {temp_path} (sync_mode: {sync_mode})")
            
        except (OSError, IOError, ValueError, TypeError) as e:
            raise AtomicJSONWriterError(f"Failed to write data to temporary file: {e}")
    
    def _atomic_replace(self, temp_path: Path, target_path: Path) -> None:
        """
        Atomically replace target file with temporary file.
        
        Args:
            temp_path: Path to temporary file
            target_path: Path to target file
            
        Raises:
            AtomicJSONWriterError: If atomic replace fails
        """
        try:
            # On Unix systems, rename is atomic if both files are on the same filesystem
            # On Windows, we need to handle the case where target exists
            if os.name == 'nt' and target_path.exists():
                # Windows doesn't allow atomic replace, so we need to remove first
                # This creates a small window of vulnerability, but it's the best we can do
                backup_path = target_path.with_suffix(target_path.suffix + '.backup')
                shutil.move(str(target_path), str(backup_path))
                try:
                    shutil.move(str(temp_path), str(target_path))
                    # Remove backup on success
                    backup_path.unlink()
                except Exception:
                    # Restore backup on failure
                    if backup_path.exists():
                        shutil.move(str(backup_path), str(target_path))
                    raise
            else:
                # Unix atomic rename
                shutil.move(str(temp_path), str(target_path))
            
            self.logger.debug(f"Atomically replaced {target_path} with {temp_path}")
            
        except OSError as e:
            raise AtomicJSONWriterError(f"Failed to atomically replace file: {e}")
    
    def _cleanup_temp_file(self, temp_path: Path) -> None:
        """
        Clean up temporary file.
        
        Args:
            temp_path: Path to temporary file to clean up
        """
        try:
            if temp_path.exists():
                temp_path.unlink()
                self.logger.debug(f"Cleaned up temporary file: {temp_path}")
        except OSError as e:
            self.logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")