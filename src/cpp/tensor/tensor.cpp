#include "tensor.h"
#include "simd/simd_config.h"
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace dllm {

Tensor::Tensor() : shape_(), dtype_(DataType::FP32) {}

Tensor::Tensor(const Shape& shape, DataType dtype)
    : shape_(shape), dtype_(dtype) {
    size_t bytes = total_bytes();
    data_ptr_ = std::malloc(bytes);
    if (!data_ptr_) {
        throw std::runtime_error("Failed to allocate tensor memory");
    }
    memset(data_ptr_, 0, bytes);
}

Tensor::Tensor(size_t batch, size_t seq, size_t hidden, DataType dtype)
    : shape_({static_cast<int64_t>(batch), static_cast<int64_t>(seq), 
              static_cast<int64_t>(hidden)}),
      dtype_(dtype) {
    size_t bytes = total_bytes();
    data_ptr_ = std::malloc(bytes);
    if (!data_ptr_) {
        throw std::runtime_error("Failed to allocate tensor memory");
    }
    memset(data_ptr_, 0, bytes);
}

Tensor::~Tensor() {
    if (data_ptr_) {
        std::free(data_ptr_);
        data_ptr_ = nullptr;
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      device_(other.device_),
      data_ptr_(other.data_ptr_),
      capacity_bytes_(other.capacity_bytes_) {
    other.data_ptr_ = nullptr;
    other.capacity_bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (data_ptr_) std::free(data_ptr_);
        
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        data_ptr_ = other.data_ptr_;
        capacity_bytes_ = other.capacity_bytes_;
        
        other.data_ptr_ = nullptr;
        other.capacity_bytes_ = 0;
    }
    return *this;
}

size_t Tensor::bytes_per_element() const {
    switch (dtype_) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::BF16: return 2;
        case DataType::INT8: return 1;
        case DataType::INT32: return 4;
    }
    return 0;
}

size_t Tensor::total_bytes() const {
    size_t elements = shape_.numel();
    return elements * bytes_per_element();
}

void* Tensor::data() { return data_ptr_; }
const void* Tensor::data() const { return data_ptr_; }

void Tensor::to(DeviceType device) {
    if (device == device_) return;
    
    // For now, CPU is the only supported device
    // GPU support requires CUDA/HIP/SYCL implementations
    if (device != DeviceType::CPU) {
        throw std::runtime_error("GPU device not available");
    }
    device_ = device;
}

void Tensor::copy_from(const Tensor& other) {
    // Use the == operator defined in Shape
    if (!(shape_ == other.shape_)) {
        throw std::runtime_error("Tensor shapes must match for copy");
    }
    if (dtype_ != other.dtype_) {
        throw std::runtime_error("Tensor dtypes must match for copy");
    }
    
    size_t bytes = total_bytes();
    memcpy(data_ptr_, other.data_ptr_, bytes);
}

void Tensor::reshape(const std::vector<int64_t>& new_dims) {
    Shape new_shape(new_dims);
    if (new_shape.numel() != shape_.numel()) {
        throw std::runtime_error("Cannot reshape tensor with different number of elements");
    }
    shape_ = new_shape;
}

Tensor Tensor::squeeze(int dim) const {
    // Not yet implemented
    return *this;
}

Tensor Tensor::unsqueeze(int dim) const {
    // Not yet implemented
    return *this;
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[" << shape_.dims[0];
    for (size_t i = 1; i < shape_.dims.size(); ++i) {
        oss << ", " << shape_.dims[i];
    }
    oss << "], dtype=" << static_cast<int>(dtype_) 
        << ", device=" << static_cast<int>(device_) << ")";
    return oss.str();
}

} // namespace dllm