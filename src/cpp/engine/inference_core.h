#pragma once

#include <string>
#include <vector>
#include <memory>

namespace dllm {

class Tensor;
class RequestHandler;

/**
 * @brief Core inference engine with SIMD optimizations
 */
class InferenceCore {
public:
    InferenceCore();
    ~InferenceCore();

    // Load model from path
    bool load_model(const std::string& model_path);

    // Run inference
    bool infer(const std::vector<int32_t>& input_tokens, 
               std::vector<float>& output);
    
    // Get model metadata
    std::string get_model_info() const;
    size_t max_context_length() const { return max_context_length_; }

private:
    size_t max_context_length_ = 2048;
    bool model_loaded_ = false;
    std::string model_path_;
};

} // namespace dllm