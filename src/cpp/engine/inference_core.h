#pragma once

#include <string>
#include <vector>
#include <memory>
#include "engine/model_loader.h"

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

    // Load model from path with format detection
    bool load_model(const std::string& model_path);
    
    // Load model with explicit format
    bool load_model(const std::string& model_path, ModelFormat format);

    // Run inference
    bool infer(const std::vector<int32_t>& input_tokens, 
               std::vector<float>& output);
    
    // Get model metadata
    std::string get_model_info() const;
    size_t max_context_length() const { return max_context_length_; }
    
    // Get loaded model metadata
    const ModelMetadata* get_metadata() const { return metadata_.get(); }

private:
    size_t max_context_length_ = 2048;
    bool model_loaded_ = false;
    std::string model_path_;
    std::unique_ptr<ModelMetadata> metadata_;
    std::unique_ptr<ModelLoader> loader_;
};

} // namespace dllm