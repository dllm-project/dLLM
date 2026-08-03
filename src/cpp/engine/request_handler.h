#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include "tensor/tensor.h"

namespace dllm {

/**
 * @brief Request types for the inference engine
 */
enum class RequestType {
    CHAT_COMPLETION = 0,
    COMPLETION = 1,
    EMBEDDING = 2,
    MODEL_INFO = 3,
    LOAD_MODEL = 4,
    UNLOAD_MODEL = 5,
};

/**
 * @brief Response structure for inference requests
 */
struct InferenceResponse {
    bool success;
    std::string error_message;
    std::vector<float> data;       // For embeddings or raw output
    std::string text;              // For text generation
    std::map<std::string, std::string> metadata;
};

/**
 * @brief Request handler for routing inference requests
 */
class RequestHandler {
public:
    RequestHandler();
    ~RequestHandler();

    // Initialize the request handler with model path
    bool initialize(const std::string& model_path);

    // Process different types of requests
    InferenceResponse handle_chat_completion(
        const std::vector<std::map<std::string, std::string>>& messages,
        float temperature = 1.0f,
        float top_p = 1.0f,
        int max_tokens = -1
    );

    InferenceResponse handle_completion(
        const std::string& prompt,
        float temperature = 1.0f,
        float top_p = 1.0f,
        int max_tokens = -1
    );

    InferenceResponse handle_embedding(const std::string& input_text);

    // Model management
    bool load_model(const std::string& model_path);
    bool unload_model();
    
    // Utility methods
    std::string get_model_info() const;
    bool is_ready() const;

private:
    // Tokenize text to tokens
    std::vector<int32_t> tokenize(const std::string& text);

    // Detokenize tokens to text
    std::string detokenize(const std::vector<int32_t>& tokens);

    // Generate inference from tokens (placeholder)
    bool generate_from_tokens(
        const std::vector<int32_t>& input_tokens,
        std::vector<float>& output,
        float temperature = 1.0f
    );

    // Format messages as text
    std::string format_messages(const std::vector<std::map<std::string, std::string>>& messages);

    bool model_loaded_ = false;
    std::string current_model_path_;
};

} // namespace dllm