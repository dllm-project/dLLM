#include "engine/model_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace dllm {

// ============================================================================
// ModelFormat utilities
// ============================================================================

std::string model_format_to_string(ModelFormat format) {
    switch (format) {
        case ModelFormat::GGUF: return "GGUF";
        case ModelFormat::SAFETENSORS: return "safetensors";
        case ModelFormat::PYTORCH: return "pytorch";
        case ModelFormat::SHARDED: return "sharded-safetensors";
        default: return "unknown";
    }
}

std::string model_format_description(ModelFormat format) {
    switch (format) {
        case ModelFormat::GGUF:
            return "GGML GGUF format - Compatible with llama.cpp, supports quantized weights (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, etc.)";
        case ModelFormat::SAFETENSORS:
            return "HuggingFace safetensors format - Single-file format with header-based tensor storage, no pickle required";
        case ModelFormat::PYTORCH:
            return "PyTorch format (.pt/.pth) - Standard PyTorch serialization format";
        case ModelFormat::SHARDED:
            return "Sharded safetensors - Multiple safetensors files for large models (>50GB)";
        default:
            return "Unknown model format";
    }
}

ModelFormat parse_model_format(const std::string& file_path) {
    std::string lower_path = file_path;
    std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);
    
    // Check for GGUF
    if (lower_path.find(".gguf") != std::string::npos) {
        return ModelFormat::GGUF;
    }
    
    // Check for safetensors
    if (lower_path.find("safetensors") != std::string::npos) {
        return ModelFormat::SAFETENSORS;
    }
    
    // Check for sharded safetensors (index.json + multiple .safetensors files)
    if (lower_path.find("model.safetensors") != std::string::npos) {
        // Check if there's an index file
        std::string index_path = file_path;
        size_t last_slash = file_path.find_last_of('/');
        if (last_slash != std::string::npos) {
            index_path = file_path.substr(0, last_slash + 1) + "model.safetensors.index.json";
        }
        if (fs::exists(index_path)) {
            return ModelFormat::SHARDED;
        }
        return ModelFormat::SAFETENSORS;
    }
    
    // Check for PyTorch
    if (lower_path.find(".pt") != std::string::npos || 
        lower_path.find(".pth") != std::string::npos) {
        return ModelFormat::PYTORCH;
    }
    
    return ModelFormat::UNKNOWN;
}

// ============================================================================
// ModelArchitecture utilities
// ============================================================================

std::string architecture_to_string(ModelArchitecture arch) {
    switch (arch) {
        case ModelArchitecture::LLAMA: return "llama";
        case ModelArchitecture::MISTRAL: return "mistral";
        case ModelArchitecture::GEMMA: return "gemma";
        case ModelArchitecture::PHI: return "phi";
        case ModelArchitecture::QWEN: return "qwen";
        case ModelArchitecture::CODELLAMA: return "codellama";
        case ModelArchitecture::CHATGLM: return "chatglm";
        case ModelArchitecture::DEEPSEEK: return "deepseek";
        case ModelArchitecture::CUSTOM: return "custom";
        default: return "unknown";
    }
}

ModelArchitecture parse_architecture(const std::string& arch_name) {
    std::string lower = arch_name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower.find("llama") != std::string::npos) return ModelArchitecture::LLAMA;
    if (lower.find("mistral") != std::string::npos) return ModelArchitecture::MISTRAL;
    if (lower.find("gemma") != std::string::npos) return ModelArchitecture::GEMMA;
    if (lower.find("phi") != std::string::npos) return ModelArchitecture::PHI;
    if (lower.find("qwen") != std::string::npos) return ModelArchitecture::QWEN;
    if (lower.find("codellama") != std::string::npos) return ModelArchitecture::CODELLAMA;
    if (lower.find("chatglm") != std::string::npos) return ModelArchitecture::CHATGLM;
    if (lower.find("deepseek") != std::string::npos) return ModelArchitecture::DEEPSEEK;
    
    return ModelArchitecture::CUSTOM;
}

// ============================================================================
// GGUF Loader (Placeholder - to be implemented with actual GGUF parsing)
// ============================================================================

class GGUFModelLoader : public ModelLoader {
public:
    GGUFModelLoader() = default;
    
    ModelMetadata load_metadata(const std::string& model_path) override {
        ModelMetadata meta;
        meta.format = ModelFormat::GGUF;
        meta.name = fs::path(model_path).stem().string();
        
        // TODO: Parse GGUF header to extract metadata
        // GGUF files contain key-value pairs in the header
        // This will be implemented when GGUF parsing library is integrated
        
        std::cout << "[GGUF] Metadata extraction placeholder for: " << model_path << std::endl;
        return meta;
    }
    
    bool load_weights(const std::string& model_path) override {
        // TODO: Implement GGUF weight loading
        // GGUF format stores tensors in a specific binary layout
        // Need to parse tensor names, shapes, and data
        
        std::cout << "[GGUF] Weight loading placeholder for: " << model_path << std::endl;
        return true;
    }
    
    bool supports(const std::string& model_path) const override {
        return parse_model_format(model_path) == ModelFormat::GGUF;
    }
    
    ModelFormat get_format() const override {
        return ModelFormat::GGUF;
    }
    
    std::string name() const override {
        return "GGUF Loader";
    }
};

// ============================================================================
// Safetensors Loader (Placeholder - to be implemented with safetensors parsing)
// ============================================================================

class SafetensorsModelLoader : public ModelLoader {
public:
    SafetensorsModelLoader() = default;
    
    ModelMetadata load_metadata(const std::string& model_path) override {
        ModelMetadata meta;
        meta.format = ModelFormat::SAFETENSORS;
        meta.name = fs::path(model_path).stem().string();
        
        // TODO: Parse safetensors header to extract metadata
        // Safetensors format has a JSON header with tensor info
        
        std::cout << "[SAFETENSORS] Metadata extraction placeholder for: " << model_path << std::endl;
        return meta;
    }
    
    bool load_weights(const std::string& model_path) override {
        // TODO: Implement safetensors weight loading
        // Safetensors format stores tensors with a JSON header followed by binary data
        
        std::cout << "[SAFETENSORS] Weight loading placeholder for: " << model_path << std::endl;
        return true;
    }
    
    bool supports(const std::string& model_path) const override {
        return parse_model_format(model_path) == ModelFormat::SAFETENSORS ||
               parse_model_format(model_path) == ModelFormat::SHARDED;
    }
    
    ModelFormat get_format() const override {
        return ModelFormat::SAFETENSORS;
    }
    
    std::string name() const override {
        return "Safetensors Loader";
    }
};

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<ModelLoader> create_model_loader(const std::string& model_path) {
    ModelFormat format = parse_model_format(model_path);
    
    switch (format) {
        case ModelFormat::GGUF:
            return std::make_unique<GGUFModelLoader>();
        case ModelFormat::SAFETENSORS:
        case ModelFormat::SHARDED:
            return std::make_unique<SafetensorsModelLoader>();
        case ModelFormat::PYTORCH:
            // PyTorch loader would be implemented similarly
            std::cerr << "[MODEL] PyTorch loader not yet implemented" << std::endl;
            return nullptr;
        default:
            std::cerr << "[MODEL] Unsupported model format: " << model_path << std::endl;
            return nullptr;
    }
}

std::vector<std::string> get_available_loaders() {
    return {
        "GGUF Loader",
        "Safetensors Loader"
    };
}

} // namespace dllm
