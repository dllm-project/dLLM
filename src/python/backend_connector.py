"""
dLLM Backend Connector
Bridges Python FastAPI with C++ inference engine via pybind11
"""

import os
from typing import List, Dict, Any

try:
    import dllm_cpp
except ImportError as e:
    print(f"Warning: Could not import dllm_cpp: {e}")
    print("Make sure the dLLM C++ library is built and installed.")
    dllm_cpp = None


class BackendConnector:
    """Connects Python API to C++ inference backend"""
    
    def __init__(self):
        self.request_handler = None
        if dllm_cpp:
            try:
                self.request_handler = dllm_cpp.RequestHandler()
            except Exception as e:
                print(f"Failed to initialize RequestHandler: {e}")
    
    def is_ready(self) -> bool:
        """Check if backend is ready for inference"""
        return self.request_handler is not None and self.request_handler.is_ready()
    
    def load_model(self, model_path: str) -> bool:
        """Load a model from path"""
        if not self.request_handler:
            raise RuntimeError("RequestHandler not available")
        return self.request_handler.load_model(model_path)
    
    def chat(self, model: str, messages: List[Dict[str, str]], 
             temperature: float = 1.0, top_p: float = 1.0,
             max_tokens: int = None) -> str:
        """Generate chat completion"""
        if not self.request_handler:
            raise RuntimeError("RequestHandler not available")
        
        # Format messages as list of dicts
        messages_list = [{"role": m["role"], "content": m["content"]} for m in messages]
        
        response = self.request_handler.handle_chat_completion(
            messages_list,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or -1
        )
        
        if not response.success:
            raise RuntimeError(f"Chat completion failed: {response.error_message}")
        
        return response.text
    
    def infer(self, model: str, prompt: str,
              temperature: float = 1.0, top_p: float = 1.0,
              max_tokens: int = None) -> str:
        """Generate completion from prompt"""
        if not self.request_handler:
            raise RuntimeError("RequestHandler not available")
        
        response = self.request_handler.handle_completion(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or -1
        )
        
        if not response.success:
            raise RuntimeError(f"Completion failed: {response.error_message}")
        
        return response.text
    
    def get_embedding(self, input_text: str, model: str) -> List[float]:
        """Get embedding for text"""
        if not self.request_handler:
            raise RuntimeError("RequestHandler not available")
        
        response = self.request_handler.handle_embedding(input_text)
        
        if not response.success:
            raise RuntimeError(f"Embedding failed: {response.error_message}")
        
        return response.data
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        # Placeholder - in production would query the loaded model or scan directory
        return [
            {"id": "llama-7b", "owned_by": "dLLM"},
            {"id": "mistral-7b", "owned_by": "dLLM"},
            {"id": "gemma-7b", "owned_by": "dLLM"}
        ]


if __name__ == "__main__":
    # Test the connector
    try:
        backend = BackendConnector()
        print(f"Backend ready: {backend.is_ready()}")
        
        if backend.is_ready():
            models = backend.list_models()
            print(f"Available models: {[m['id'] for m in models]}")
    except Exception as e:
        print(f"Test failed: {e}")