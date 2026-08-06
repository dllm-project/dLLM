#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>
#include <stdexcept>

namespace dllm {

/**
 * @brief Supported model weight file formats
 */
enum class ModelFormat {
    UNKNOWN = 0,
    GGUF = 1,       // GGML GGUF format (llama.cpp compatible)
    SAFETENSORS = 2, // HuggingFace safetensors format
    PYTORCH = 3,     // PyTorch .pt/.pth format
    SHARDED = 4,     // Sharded safetensors (multiple files)
};

/**
 * @brief Convert ModelFormat enum to string
 */
std::string model_format_to_string(ModelFormat format);

/**
 * @brief Parse ModelFormat from file extension
 */
ModelFormat parse_model_format(const std::string& file_path);

/**
 * @brief Get human-readable description of a model format
 */
std::string model_format_description(ModelFormat format);

/**
 * @brief Model architecture type
 */
enum class ModelArchitecture {
    UNKNOWN = 0,
    LLAMA = 1,
    MISTRAL = 2,
    GEMMA = 3,
    PHI = 4,
    QWEN = 5,
    CODELLAMA = 6,
    CHATGLM = 7,
    DEEPSEEK = 8,
    CUSTOM = 9,
};

/**
 * @brief Convert ModelArchitecture enum to string
 */
std::string architecture_to_string(ModelArchitecture arch);

/**
 * @brief Parse ModelArchitecture from model metadata
 */
ModelArchitecture parse_architecture(const std::string& arch_name);

/**
 * @brief Model metadata extracted from weight files
 */
struct ModelMetadata {
    std::string name;
    std::string description;
    ModelFormat format;
    ModelArchitecture architecture;
    size_t num_parameters;
    size_t num_layers;
    size_t vocab_size;
    size_t hidden_size;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t max_position_embeddings;
    float rope_theta;
    float rms_norm_eps;
    bool is_chat_model;
    std::map<std::string, std::string> custom_metadata;

    ModelMetadata()
        : format(ModelFormat::UNKNOWN),
          architecture(ModelArchitecture::UNKNOWN),
          num_parameters(0),
          num_layers(0),
          vocab_size(0),
          hidden_size(0),
          num_attention_heads(0),
          num_key_value_heads(0),
          max_position_embeddings(0),
          rope_theta(10000.0f),
          rms_norm_eps(1e-5f),
          is_chat_model(false) {}
};

/**
 * @brief Abstract base class for model format loaders
 */
class ModelLoader {
public:
    virtual ~ModelLoader() = default;

    /**
     * @brief Load model metadata from file
     */
    virtual ModelMetadata load_metadata(const std::string& model_path) = 0;

    /**
     * @brief Load model weights into tensors
     */
    virtual bool load_weights(const std::string& model_path) = 0;

    /**
     * @brief Check if this loader supports the given file
     */
    virtual bool supports(const std::string& model_path) const = 0;

    /**
     * @brief Get the format this loader handles
     */
    virtual ModelFormat get_format() const = 0;

    /**
     * @brief Get loader name for logging
     */
    virtual std::string name() const = 0;
};

/**
 * @brief Factory function to create appropriate ModelLoader for a file
 */
std::unique_ptr<ModelLoader> create_model_loader(const std::string& model_path);

/**
 * @brief Get all registered model loaders
 */
std::vector<std::string> get_available_loaders();

} // namespace dllm
