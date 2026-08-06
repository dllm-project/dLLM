#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

namespace dllm {

/**
 * @brief Data types supported by the tensor
 */
enum class DataType {
    FP32 = 0,   // 32-bit floating point
    FP16 = 1,   // 16-bit floating point (half precision)
    BF16 = 2,   // 16-bit bfloat (brain float)
    INT8 = 3,   // 8-bit integer
    INT32 = 4,  // 32-bit integer
};

/**
 * @brief Convert DataType to string representation
 */
std::string data_type_to_string(DataType dtype);

/**
 * @brief Convert string to DataType
 */
DataType string_to_data_type(const std::string& str);

/**
 * @brief Device type for tensor storage
 */
enum class DeviceType {
    CPU = 0,
    CUDA = 1,
    HIP = 2,
    SYCL = 3,
};

/**
 * @brief Tensor shape representation
 */
struct Shape {
    std::vector<int64_t> dims;
    
    Shape() : dims{} {}
    explicit Shape(const std::vector<int64_t>& d) : dims(d) {}
    explicit Shape(std::initializer_list<int64_t> init) : dims(init) {}
    
    size_t rank() const { return dims.size(); }
    size_t numel() const {
        size_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    int64_t dim(int i) const { return dims.at(i); }
    
    bool operator==(const Shape& other) const {
        if (dims.size() != other.dims.size()) return false;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] != other.dims[i]) return false;
        }
        return true;
    }
};

/**
 * @brief Core tensor class with memory management
 */
class Tensor {
public:
    // Constructor/Destructor
    Tensor();
    Tensor(const Shape& shape, DataType dtype = DataType::FP32);
    Tensor(size_t batch, size_t seq, size_t hidden, DataType dtype = DataType::FP32);
    ~Tensor();

    // Copy and move semantics
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Accessors
    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    DeviceType device() const { return device_; }
    
    size_t bytes_per_element() const;
    size_t total_bytes() const;
    
    void* data();
    const void* data() const;

    // Device management
    void to(DeviceType device);
    void copy_from(const Tensor& other);

    // Shape operations
    void reshape(const std::vector<int64_t>& new_dims);
    Tensor& squeeze(int dim = -1);
    Tensor& unsqueeze(int dim = -1);

    // Utility
    std::string to_string() const;

    // Model weight loading support
    bool load_from_file(const std::string& file_path, const std::string& tensor_name = "");
    bool save_to_file(const std::string& file_path, const std::string& tensor_name = "") const;
    size_t load_from_buffer(const void* data, size_t size);

private:
    Shape shape_;
    DataType dtype_;
    DeviceType device_ = DeviceType::CPU;
    
    void* data_ptr_ = nullptr;
    size_t capacity_bytes_ = 0;
};

} // namespace dllm