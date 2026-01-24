# -*- coding=utf-8 -*-
"""
Graceful Shutdown Handler

This module provides signal handling capabilities to ensure that in-progress
operations can complete before the program terminates. It tracks ongoing
operations and provides a mechanism to wait for their completion during shutdown.
"""

import signal
import threading
import time
import logging
from typing import Set, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PendingOperation:
    """Represents a pending operation that should complete before shutdown."""
    operation_id: str
    start_time: float
    operation_type: str
    target_file: str


class GracefulShutdownHandler:
    """
    Handles graceful shutdown by tracking ongoing operations and ensuring
    they complete before the program terminates.
    
    This class registers signal handlers for SIGTERM and SIGINT, and provides
    a mechanism to track operations that should complete before shutdown.
    """
    
    _instance: Optional['GracefulShutdownHandler'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'GracefulShutdownHandler':
        """Ensure singleton pattern for signal handler."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        """Initialize the graceful shutdown handler."""
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.logger = logging.getLogger(__name__)
        self._pending_operations: Dict[str, PendingOperation] = {}
        self._operations_lock = threading.RLock()
        self._shutdown_requested = False
        self._shutdown_event = threading.Event()
        self._original_handlers: Dict[int, Any] = {}
        
        # Register signal handlers
        self._register_signal_handlers()
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        try:
            # Store original handlers
            self._original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._signal_handler)
            self._original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._signal_handler)
            
            self.logger.debug("Registered signal handlers for graceful shutdown")
            
        except (OSError, ValueError) as e:
            self.logger.warning(f"Failed to register signal handlers: {e}")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle termination signals gracefully.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        self.logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
        
        with self._operations_lock:
            self._shutdown_requested = True
            
            if not self._pending_operations:
                self.logger.info("No pending operations, shutting down immediately")
                self._shutdown_event.set()
                self._restore_signal_handlers()
                # Re-raise the signal to allow normal termination
                signal.signal(signum, signal.SIG_DFL)
                signal.raise_signal(signum)
                return
        
        # Start shutdown process in a separate thread to avoid blocking signal handler
        shutdown_thread = threading.Thread(
            target=self._handle_graceful_shutdown,
            args=(signum,),
            daemon=True
        )
        shutdown_thread.start()
    
    def _handle_graceful_shutdown(self, signum: int) -> None:
        """
        Handle the graceful shutdown process.
        
        Args:
            signum: Signal number that triggered shutdown
        """
        self.logger.info(f"Starting graceful shutdown process for {len(self._pending_operations)} operations")
        
        # Wait for operations to complete with timeout
        success = self.wait_for_operations(timeout=30.0)
        
        if success:
            self.logger.info("All operations completed successfully, shutting down")
        else:
            self.logger.warning("Timeout waiting for operations to complete, forcing shutdown")
        
        self._shutdown_event.set()
        self._restore_signal_handlers()
        
        # Re-raise the signal to allow normal termination
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)
    
    def register_operation(self, operation_id: str, operation_type: str = "unknown", target_file: str = "") -> None:
        """
        Register an ongoing operation that should complete before shutdown.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation (e.g., "json_write")
            target_file: Target file path for the operation
        """
        with self._operations_lock:
            if self._shutdown_requested:
                self.logger.warning(f"Shutdown already requested, but registering operation {operation_id}")
            
            operation = PendingOperation(
                operation_id=operation_id,
                start_time=time.time(),
                operation_type=operation_type,
                target_file=target_file
            )
            
            self._pending_operations[operation_id] = operation
            self.logger.debug(f"Registered operation: {operation_id} ({operation_type})")
    
    def unregister_operation(self, operation_id: str) -> None:
        """
        Unregister a completed operation.
        
        Args:
            operation_id: Unique identifier for the operation
        """
        with self._operations_lock:
            if operation_id in self._pending_operations:
                operation = self._pending_operations.pop(operation_id)
                duration = time.time() - operation.start_time
                self.logger.debug(f"Unregistered operation: {operation_id} (completed in {duration:.2f}s)")
            else:
                self.logger.warning(f"Attempted to unregister unknown operation: {operation_id}")
    
    def wait_for_operations(self, timeout: float = 30.0) -> bool:
        """
        Wait for all pending operations to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if all operations completed, False if timeout occurred
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._operations_lock:
                if not self._pending_operations:
                    return True
                
                # Log pending operations periodically
                if int(time.time() - start_time) % 5 == 0:
                    self.logger.info(f"Waiting for {len(self._pending_operations)} operations to complete...")
                    for op_id, operation in self._pending_operations.items():
                        elapsed = time.time() - operation.start_time
                        self.logger.debug(f"  - {op_id} ({operation.operation_type}): {elapsed:.1f}s")
            
            time.sleep(0.1)
        
        # Timeout occurred
        with self._operations_lock:
            if self._pending_operations:
                self.logger.error(f"Timeout waiting for {len(self._pending_operations)} operations:")
                for op_id, operation in self._pending_operations.items():
                    elapsed = time.time() - operation.start_time
                    self.logger.error(f"  - {op_id} ({operation.operation_type}): {elapsed:.1f}s")
                return False
        
        return True
    
    def is_shutdown_requested(self) -> bool:
        """
        Check if shutdown has been requested.
        
        Returns:
            bool: True if shutdown was requested
        """
        return self._shutdown_requested
    
    def get_pending_operations(self) -> Dict[str, PendingOperation]:
        """
        Get a copy of currently pending operations.
        
        Returns:
            Dict[str, PendingOperation]: Copy of pending operations
        """
        with self._operations_lock:
            return self._pending_operations.copy()
    
    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        try:
            for signum, handler in self._original_handlers.items():
                if handler is not None:
                    signal.signal(signum, handler)
            self.logger.debug("Restored original signal handlers")
        except (OSError, ValueError) as e:
            self.logger.warning(f"Failed to restore signal handlers: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources and restore signal handlers."""
        self._restore_signal_handlers()
        with self._operations_lock:
            self._pending_operations.clear()
        self.logger.debug("Graceful shutdown handler cleaned up")


# Global instance for easy access
_shutdown_handler: Optional[GracefulShutdownHandler] = None


def get_shutdown_handler() -> GracefulShutdownHandler:
    """
    Get the global shutdown handler instance.
    
    Returns:
        GracefulShutdownHandler: The global shutdown handler
    """
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdownHandler()
    return _shutdown_handler


def register_operation(operation_id: str, operation_type: str = "unknown", target_file: str = "") -> None:
    """
    Convenience function to register an operation with the global shutdown handler.
    
    Args:
        operation_id: Unique identifier for the operation
        operation_type: Type of operation
        target_file: Target file path for the operation
    """
    get_shutdown_handler().register_operation(operation_id, operation_type, target_file)


def unregister_operation(operation_id: str) -> None:
    """
    Convenience function to unregister an operation with the global shutdown handler.
    
    Args:
        operation_id: Unique identifier for the operation
    """
    get_shutdown_handler().unregister_operation(operation_id)