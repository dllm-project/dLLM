"""
dLLM Backend Connector

Provides a Python interface to the C++ backend inference engine.
"""

import os
from enum import Enum
from typing import Optional, List, Dict, Any


class ModelFormat(Enum):
    """Supported model formats."""
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    SHARDED_SAFETENSORS = "sharded_safetensors"
    PYTORCH = "pytorch"


class BackendConnector:
    """
    Connector to the C++ backend inference engine.
    
    This class provides methods to load models, run inference,
    and manage the backend engine.
    """
    
    def __init__(self, cpp_library_path: Optional[str] = None):
        """
        Initialize the backend connector.
        
        Args:
            cpp_library_path: Path to the C++ shared library.
                             If None, will look for libdllm.so in standard locations.
        """
        self._library_path = cpp_library_path or self._find_library()
        self._model_loaded = False
        self._model_metadata = None
        
    def _find_library(self) -> str:
        """Find the C++ library in standard locations."""
        search_paths = [
            os.path.join(os.path.dirname(__file__), "..", "build"),
            os.path.join(os.getcwd(), "build"),
            "/usr/local/lib",
            "/usr/lib",
        ]
        
        for path in search_paths:
            lib_path = os.path.join(path, "libdllm.so")
            if os.path.exists(lib_path):
                return lib_path
                
        raise FileNotFoundError(
            f"Could not find libdllm.so. Searched in: {search_paths}"
        )
    
    def load_model(
        self,
        model_path: str,
        model_format: Optional[ModelFormat] = None
    ) -> Dict[str, Any]:
        """
        Load a model from the specified path.
        
        Args:
            model_path: Path to the model file
            model_format: Model format (auto-detected if None)
            
        Returns:
            Model metadata dictionary
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is invalid
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Auto-detect format if not specified
        if model_format is None:
            model_format = self._detect_format(model_path)
            
        # Load model via C++ backend
        try:
            import dllm_cpp
            self._model_metadata = dllm_cpp.load_model(model_path, model_format.value)
            self._model_loaded = True
            return self._model_metadata
        except ImportError:
            # Fallback to Python-only mode
            self._model_metadata = {
                "path": model_path,
                "format": model_format.value,
                "loaded": True,
            }
            self._model_loaded = True
            return self._model_metadata
    
    def _detect_format(self, model_path: str) -> ModelFormat:
        """Detect model format from file extension."""
        ext = os.path.splitext(model_path)[1].lower()
        
        format_map = {
            ".gguf": ModelFormat.GGUF,
            ".safetensors": ModelFormat.SAFETENSORS,
            ".pt": ModelFormat.PYTORCH,
            ".pth": ModelFormat.PYTORCH,
        }
        
        if ext in format_map:
            return format_map[ext]
            
        raise ValueError(f"Unsupported model format: {ext}")
    
    def is_ready(self) -> bool:
        """Check if a model is loaded and ready for inference."""
        return self._model_loaded
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 256,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate chat completions.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Chat completion response dictionary
        """
        if not self._model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
            
        # Generate response via C++ backend
        try:
            import dllm_cpp
            response = dllm_cpp.chat(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=stream
            )
            return response
        except ImportError:
            # Fallback response
            return {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": int(__import__("time").time()),
                "model": self._model_metadata.get("name", "unknown"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[C++ backend not available - using fallback]"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
    
    def infer(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text completions.
        
        Args:
            prompt: Input text prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Completion response dictionary
        """
        if not self._model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
            
        try:
            import dllm_cpp
            return dllm_cpp.infer(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        except ImportError:
            return {
                "id": "cmpl-local",
                "object": "text_completion",
                "created": int(__import__("time").time()),
                "model": self._model_metadata.get("name", "unknown"),
                "choices": [{
                    "text": "[C++ backend not available - using fallback]",
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
    
    def get_embedding(
        self,
        text: str,
        model: str = "default"
    ) -> List[float]:
        """
        Generate embedding for input text.
        
        Args:
            text: Input text
            model: Model to use for embedding
            
        Returns:
            Embedding vector as list of floats
        """
        if not self._model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
            
        try:
            import dllm_cpp
            return dllm_cpp.get_embedding(text, model)
        except ImportError:
            # Return zero vector as fallback
            return [0.0] * 768
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of model metadata dictionaries
        """
        if self._model_loaded and self._model_metadata:
            return [self._model_metadata]
        return []
    
    def unload_model(self):
        """Unload the current model."""
        try:
            import dllm_cpp
            dllm_cpp.unload_model()
        except ImportError:
            pass
        finally:
            self._model_loaded = False
            self._model_metadata = None
