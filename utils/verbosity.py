import os
import sys
import logging
from contextlib import contextmanager
from typing import Optional, TextIO

class VerbosityManager:
    """Manages verbosity levels across the entire application"""
    
    QUIET = 0      # Essential info only
    NORMAL = 1     # Default balance
    VERBOSE = 2    # Full debugging
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VerbosityManager()
        return cls._instance
    
    def __init__(self):
        self.level = self.NORMAL
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._null_device = open(os.devnull, 'w')
        self._suppress_model_output = False
        
    def set_level(self, level: int):
        """Set verbosity level"""
        self.level = level
        
        # Configure logging level based on verbosity
        if level == self.QUIET:
            logging.getLogger().setLevel(logging.WARNING)
            # Also set environment variable for llama.cpp
            os.environ["LLAMA_VERBOSE"] = "0"
        elif level == self.VERBOSE:
            logging.getLogger().setLevel(logging.DEBUG)
            os.environ["LLAMA_VERBOSE"] = "2"
        else:  # NORMAL
            logging.getLogger().setLevel(logging.INFO)
            os.environ["LLAMA_VERBOSE"] = "1"
            
    def set_suppress_model_output(self, suppress: bool):
        """Control whether to suppress model output"""
        self._suppress_model_output = suppress
    
    @contextmanager
    def suppress_output(self, suppress: bool = True):
        """Context manager to temporarily suppress stdout/stderr"""
        if suppress:
            sys.stdout = self._null_device
            sys.stderr = self._null_device
            try:
                yield
            finally:
                sys.stdout = self._original_stdout
                sys.stderr = self._original_stderr
        else:
            yield
    
    def get_llm_config(self) -> dict:
        """Get configuration for LLM verbosity"""
        return {
            "verbose": self.level > self.QUIET and not self._suppress_model_output
        }
    
    def cleanup(self):
        """Clean up resources"""
        # Restore stdout/stderr if they were changed
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
        # Close null device
        self._null_device.close()
        
    def silence_all(self):
        """Completely silence all output"""
        # Redirect stdout and stderr to null
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        # Set environment variables to control external libraries
        os.environ["LLAMA_VERBOSE"] = "0"
        
        # Set Python logging to critical only
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Flag for complete silence
        self.completely_silent = True