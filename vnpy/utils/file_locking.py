# -*- coding=utf-8 -*-
"""
File Locking Utilities

This module provides cross-platform file locking capabilities to prevent
concurrent writes to the same JSON file during atomic operations.
"""

import os
import time
import logging
import threading
from typing import Optional, Dict, Any
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass

# Platform-specific imports
try:
    import fcntl  # Unix/Linux/macOS
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt  # Windows
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


@dataclass
class LockInfo:
    """Information about a file lock."""
    filepath: str
    lock_type: str
    acquired_time: float
    process_id: int
    thread_id: int


class FileLockError(Exception):
    """Base exception for file locking errors."""
    pass


class LockTimeoutError(FileLockError):
    """Raised when lock acquisition times out."""
    pass


class LockAcquisitionError(FileLockError):
    """Raised when lock cannot be acquired."""
    pass


class CrossPlatformFileLock:
    """
    Cross-platform file locking implementation.
    
    Provides exclusive file locking to prevent concurrent writes to the same
    file across processes and threads.
    """
    
    def __init__(self, filepath: str, timeout: float = 30.0):
        """
        Initialize the file lock.
        
        Args:
            filepath: Path to the file to lock
            timeout: Maximum time to wait for lock acquisition
        """
        self.filepath = Path(filepath).resolve()
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Lock file path (adjacent to target file)
        self.lock_filepath = self.filepath.with_suffix(self.filepath.suffix + '.lock')
        
        # Internal state
        self._lock_file: Optional[Any] = None
        self._acquired = False
        self._lock_info: Optional[LockInfo] = None
    
    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire the file lock.
        
        Args:
            blocking: Whether to block waiting for the lock
            
        Returns:
            bool: True if lock was acquired successfully
            
        Raises:
            LockTimeoutError: If timeout occurs while waiting for lock
            LockAcquisitionError: If lock cannot be acquired
        """
        if self._acquired:
            self.logger.warning(f"Lock already acquired for {self.filepath}")
            return True
        
        start_time = time.time()
        
        while True:
            try:
                # Try to acquire the lock
                if self._try_acquire_lock():
                    self._acquired = True
                    self._lock_info = LockInfo(
                        filepath=str(self.filepath),
                        lock_type="exclusive",
                        acquired_time=time.time(),
                        process_id=os.getpid(),
                        thread_id=threading.get_ident()
                    )
                    
                    self.logger.debug(f"Acquired lock for {self.filepath}")
                    return True
                
                # If non-blocking, return immediately
                if not blocking:
                    return False
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.timeout:
                    raise LockTimeoutError(
                        f"Timeout waiting for lock on {self.filepath} after {elapsed:.2f}s"
                    )
                
                # Wait a bit before retrying
                time.sleep(0.01)
                
            except Exception as e:
                if isinstance(e, (LockTimeoutError, LockAcquisitionError)):
                    raise
                raise LockAcquisitionError(f"Failed to acquire lock for {self.filepath}: {e}")
    
    def release(self) -> bool:
        """
        Release the file lock.
        
        Returns:
            bool: True if lock was released successfully
        """
        if not self._acquired:
            self.logger.debug(f"Lock not acquired for {self.filepath}, nothing to release")
            return True
        
        try:
            self._release_lock()
            self._acquired = False
            self._lock_info = None
            
            self.logger.debug(f"Released lock for {self.filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to release lock for {self.filepath}: {e}")
            return False
    
    def _try_acquire_lock(self) -> bool:
        """
        Platform-specific lock acquisition.
        
        Returns:
            bool: True if lock was acquired
        """
        try:
            # Create lock file
            self._lock_file = open(self.lock_filepath, 'w')
            
            if HAS_FCNTL:
                # Unix/Linux/macOS
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif HAS_MSVCRT:
                # Windows
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                # Fallback: use file existence as lock (not perfect but better than nothing)
                if self.lock_filepath.exists():
                    self._lock_file.close()
                    return False
            
            # Write lock info to file
            lock_data = {
                'pid': os.getpid(),
                'thread_id': threading.get_ident(),
                'timestamp': time.time(),
                'target_file': str(self.filepath)
            }
            
            import json
            json.dump(lock_data, self._lock_file, indent=2)
            self._lock_file.flush()
            
            return True
            
        except (OSError, IOError):
            # Lock is held by another process
            if self._lock_file:
                try:
                    self._lock_file.close()
                except:
                    pass
                self._lock_file = None
            return False
    
    def _release_lock(self) -> None:
        """Platform-specific lock release."""
        if self._lock_file:
            try:
                if HAS_FCNTL:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                elif HAS_MSVCRT:
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                
                self._lock_file.close()
            except Exception as e:
                self.logger.warning(f"Error during lock release: {e}")
            finally:
                self._lock_file = None
        
        # Remove lock file
        try:
            if self.lock_filepath.exists():
                self.lock_filepath.unlink()
        except OSError as e:
            self.logger.warning(f"Failed to remove lock file {self.lock_filepath}: {e}")
    
    def is_locked(self) -> bool:
        """
        Check if the lock is currently held.
        
        Returns:
            bool: True if lock is held
        """
        return self._acquired
    
    def get_lock_info(self) -> Optional[LockInfo]:
        """
        Get information about the current lock.
        
        Returns:
            Optional[LockInfo]: Lock information if lock is held
        """
        return self._lock_info
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor - ensure lock is released."""
        if self._acquired:
            self.release()


class FileLockManager:
    """
    Manager for handling multiple file locks and preventing deadlocks.
    """
    
    def __init__(self):
        """Initialize the lock manager."""
        self.logger = logging.getLogger(__name__)
        self._active_locks: Dict[str, CrossPlatformFileLock] = {}
        self._lock_registry_mutex = threading.RLock()
    
    @contextmanager
    def acquire_lock(self, filepath: str, timeout: float = 30.0):
        """
        Context manager for acquiring and releasing file locks.
        
        Args:
            filepath: Path to file to lock
            timeout: Lock acquisition timeout
            
        Yields:
            CrossPlatformFileLock: The acquired lock
            
        Raises:
            LockTimeoutError: If lock acquisition times out
            LockAcquisitionError: If lock cannot be acquired
        """
        filepath = str(Path(filepath).resolve())
        
        with self._lock_registry_mutex:
            # Check if we already have a lock for this file in this process
            if filepath in self._active_locks:
                existing_lock = self._active_locks[filepath]
                if existing_lock.is_locked():
                    self.logger.debug(f"Reusing existing lock for {filepath}")
                    yield existing_lock
                    return
            
            # Create new lock
            file_lock = CrossPlatformFileLock(filepath, timeout)
            
            try:
                # Acquire the lock
                file_lock.acquire()
                self._active_locks[filepath] = file_lock
                
                self.logger.debug(f"Acquired new lock for {filepath}")
                yield file_lock
                
            finally:
                # Always release and clean up
                file_lock.release()
                if filepath in self._active_locks:
                    del self._active_locks[filepath]
    
    def cleanup_stale_locks(self, max_age_hours: int = 24) -> int:
        """
        Clean up stale lock files that may have been left behind.
        
        Args:
            max_age_hours: Maximum age of lock files to keep
            
        Returns:
            int: Number of stale locks cleaned up
        """
        cleaned_count = 0
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # Find all .lock files in common directories
        search_paths = [
            Path.cwd(),
            Path.home() / '.vnpy',
            Path('/tmp') if Path('/tmp').exists() else None
        ]
        
        for search_path in search_paths:
            if not search_path or not search_path.exists():
                continue
                
            try:
                for lock_file in search_path.glob('**/*.lock'):
                    try:
                        # Check if lock file is stale
                        stat = lock_file.stat()
                        if stat.st_mtime < cutoff_time:
                            # Try to read lock info
                            try:
                                with open(lock_file, 'r') as f:
                                    import json
                                    lock_data = json.load(f)
                                    
                                # Check if the process that created the lock is still running
                                if self._is_process_running(lock_data.get('pid')):
                                    continue  # Process still running, don't remove
                                    
                            except (json.JSONDecodeError, OSError):
                                pass  # Invalid lock file, safe to remove
                            
                            # Remove stale lock file
                            lock_file.unlink()
                            cleaned_count += 1
                            self.logger.info(f"Cleaned up stale lock file: {lock_file}")
                            
                    except OSError as e:
                        self.logger.warning(f"Failed to process lock file {lock_file}: {e}")
                        
            except OSError as e:
                self.logger.warning(f"Failed to search for lock files in {search_path}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} stale lock files")
        
        return cleaned_count
    
    def _is_process_running(self, pid: Optional[int]) -> bool:
        """
        Check if a process with the given PID is still running.
        
        Args:
            pid: Process ID to check
            
        Returns:
            bool: True if process is running
        """
        if not pid:
            return False
            
        try:
            # Send signal 0 to check if process exists (Unix/Linux/macOS)
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
        except AttributeError:
            # Windows doesn't have os.kill, use alternative method
            try:
                import psutil
                return psutil.pid_exists(pid)
            except ImportError:
                # If psutil not available, assume process might be running
                return True
    
    def get_active_locks(self) -> Dict[str, LockInfo]:
        """
        Get information about currently active locks.
        
        Returns:
            Dict[str, LockInfo]: Active locks by filepath
        """
        with self._lock_registry_mutex:
            result = {}
            for filepath, lock in self._active_locks.items():
                if lock.is_locked():
                    lock_info = lock.get_lock_info()
                    if lock_info:
                        result[filepath] = lock_info
            return result


# Global lock manager instance
_lock_manager: Optional[FileLockManager] = None


def get_lock_manager() -> FileLockManager:
    """
    Get the global file lock manager instance.
    
    Returns:
        FileLockManager: The global lock manager
    """
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = FileLockManager()
    return _lock_manager


def acquire_file_lock(filepath: str, timeout: float = 30.0):
    """
    Context manager for acquiring a file lock.
    
    Args:
        filepath: Path to file to lock
        timeout: Lock acquisition timeout
        
    Returns:
        Context manager for the file lock
    """
    return get_lock_manager().acquire_lock(filepath, timeout)