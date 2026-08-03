#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tensor/tensor.h"
#include "simd/simd_ops.h"
#include "engine/inference_core.h"
#include "engine/request_handler.h"

namespace py = pybind11;

/**
 * @brief Python bridge for dLLM C++ library
 */

PYBIND11_MODULE(dllm_cpp, m) {
    m.doc() = "dLLM - Distributed CPU AI Inference Engine with SIMD optimizations";
    
    // Expose DataType enum
    py::enum_<dllm::DataType>(m, "DataType")
        .value("FP32", dllm::DataType::FP32)
        .value("FP16", dllm::DataType::FP16)
        .value("BF16", dllm::DataType::BF16)
        .value("INT8", dllm::DataType::INT8)
        .value("INT32", dllm::DataType::INT32)
        .export_values();
    
    // Expose DeviceType enum
    py::enum_<dllm::DeviceType>(m, "DeviceType")
        .value("CPU", dllm::DeviceType::CPU)
        .value("CUDA", dllm::DeviceType::CUDA)
        .value("HIP", dllm::DeviceType::HIP)
        .value("SYCL", dllm::DeviceType::SYCL)
        .export_values();
    
    // Expose Shape struct
    py::class_<dllm::Shape>(m, "Shape")
        .def(py::init<>())
        .def(py::init<const std::vector<int64_t>&>())
        .def("rank", &dllm::Shape::rank)
        .def("numel", &dllm::Shape::numel)
        .def("dim", &dllm::Shape::dim)
        .def("__repr__", [](const dllm::Shape& s) {
            std::ostringstream oss;
            oss << "Shape(" << s.dims << ")";
            return oss.str();
        });
    
    // Expose Tensor class
    py::class_<dllm::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const dllm::Shape&, dllm::DataType>(), 
             py::arg("shape"), py::arg("dtype") = dllm::DataType::FP32)
        .def(py::init<size_t, size_t, size_t, dllm::DataType>(),
             py::arg("batch"), py::arg("seq"), py::arg("hidden"), 
             py::arg("dtype") = dllm::DataType::FP32)
        
        // Properties
        .def_property_readonly("shape", &dllm::Tensor::shape)
        .def_property_readonly("dtype", &dllm::Tensor::dtype)
        .def_property_readonly("device", &dllm::Tensor::device)
        .def_property_readonly("bytes_per_element", &dllm::Tensor::bytes_per_element)
        .def_property_readonly("total_bytes", &dllm::Tensor::total_bytes)
        
        // Methods
        .def("data_ptr", [](const dllm::Tensor& t) -> py::buffer_info {
            void* ptr = const_cast<void*>(t.data());
            return py::buffer_info(
                ptr,
                sizeof(float),
                py::format_descriptor<float>::format(),
                1,
                {static_cast<size_t>(t.shape().numel())},
                {sizeof(float)}
            );
        })
        
        .def("to", &dllm::Tensor::to)
        .def("copy_from", &dllm::Tensor::copy_from)
        .def("reshape", &dllm::Tensor::reshape)
        .def("__repr__", &dllm::Tensor::to_string);
    
    // SIMD functions - Vector operations
    m.def("vector_add_sse42", &dllm::simd::vector_add_sse42,
          "SSE4.2 optimized vector addition");
    m.def("vector_add_avx", &dllm::simd::vector_add_avx,
          "AVX optimized vector addition");
    m.def("vector_add_avx2", &dllm::simd::vector_add_avx2,
          "AVX2 optimized vector addition");
    
    m.def("vector_sub_sse42", &dllm::simd::vector_sub_sse42,
          "SSE4.2 optimized vector subtraction");
    m.def("vector_sub_avx", &dllm::simd::vector_sub_avx,
          "AVX optimized vector subtraction");
    m.def("vector_sub_avx2", &dllm::simd::vector_sub_avx2,
          "AVX2 optimized vector subtraction");
    
    m.def("vector_scale_sse42", &dllm::simd::vector_scale_sse42,
          "SSE4.2 optimized scalar-vector multiplication");
    m.def("vector_scale_avx", &dllm::simd::vector_scale_avx,
          "AVX optimized scalar-vector multiplication");
    m.def("vector_scale_avx2", &dllm::simd::vector_scale_avx2,
          "AVX2 optimized scalar-vector multiplication");
    
    m.def("vector_dot_sse42", &dllm::simd::vector_dot_sse42,
          "SSE4.2 optimized dot product");
    m.def("vector_dot_avx", &dllm::simd::vector_dot_avx,
          "AVX optimized dot product");
    m.def("vector_dot_avx2", &dllm::simd::vector_dot_avx2,
          "AVX2 optimized dot product with FMA");
    
    m.def("vector_sum_sse42", &dllm::simd::vector_sum_sse42,
          "SSE4.2 optimized vector sum");
    m.def("vector_sum_avx", &dllm::simd::vector_sum_avx,
          "AVX optimized vector sum");
    m.def("vector_sum_avx2", &dllm::simd::vector_sum_avx2,
          "AVX2 optimized vector sum with FMA");
    
    // GEMM
    m.def("gemm_avx2", &dllm::simd::gemm_avx2,
          "AVX2 optimized General Matrix Multiply");
    
    // Activation functions
    m.def("relu_avx2", &dllm::simd::relu_avx2,
          "AVX2 optimized ReLU activation");
    m.def("leaky_relu_avx2", &dllm::simd::leaky_relu_avx2,
          "AVX2 optimized Leaky ReLU activation");
    m.def("gelu_avx2", &dllm::simd::gelu_avx2,
          "AVX2 optimized GELU activation");
    m.def("silu_avx2", &dllm::simd::silu_avx2,
          "AVX2 optimized SiLU (Swish) activation");
    
    // Utility
    m.def("get_instruction_set", []() {
        return static_cast<int>(dllm::simd::get_current_instruction_set());
    }, "Get the current working instruction set level");

    // ============================================
    // InferenceCore class bindings
    // ============================================
    py::class_<dllm::InferenceCore>(m, "InferenceCore")
        .def(py::init<>())
        .def("load_model", &dllm::InferenceCore::load_model,
             "Load model from path")
        .def("infer", &dllm::InferenceCore::infer,
             "Run inference on input tokens",
             py::arg("input_tokens"), py::arg("output"))
        .def("get_model_info", &dllm::InferenceCore::get_model_info,
             "Get model information")
        .def_property_readonly("max_context_length", 
                               &dllm::InferenceCore::max_context_length);

    // ============================================
    // RequestHandler class bindings
    // ============================================
    py::class_<dllm::RequestHandler>(m, "RequestHandler")
        .def(py::init<>())
        .def("initialize", &dllm::RequestHandler::initialize,
             "Initialize with model path")
        .def("load_model", &dllm::RequestHandler::load_model,
             "Load model from path")
        .def("unload_model", &dllm::RequestHandler::unload_model,
             "Unload current model")
        .def("handle_chat_completion", &dllm::RequestHandler::handle_chat_completion,
             "Process chat completion request",
             py::arg("messages"), py::arg("temperature") = 1.0f,
             py::arg("top_p") = 1.0f, py::arg("max_tokens") = -1)
        .def("handle_completion", &dllm::RequestHandler::handle_completion,
             "Process completion request",
             py::arg("prompt"), py::arg("temperature") = 1.0f,
             py::arg("top_p") = 1.0f, py::arg("max_tokens") = -1)
        .def("handle_embedding", &dllm::RequestHandler::handle_embedding,
             "Process embedding request",
             py::arg("input_text"))
        .def("get_model_info", &dllm::RequestHandler::get_model_info,
             "Get model information")
        .def("is_ready", &dllm::RequestHandler::is_ready,
             "Check if engine is ready");

    // ============================================
    // InferenceResponse bindings
    // ============================================
    py::class_<dllm::InferenceResponse>(m, "InferenceResponse")
        .def(py::init<>())
        .def_readwrite("success", &dllm::InferenceResponse::success)
        .def_readwrite("error_message", &dllm::InferenceResponse::error_message)
        .def_readwrite("data", &dllm::InferenceResponse::data)
        .def_readwrite("text", &dllm::InferenceResponse::text);

    // Expose InstructionSet enum
    py::enum_<dllm::simd::InstructionSet>(m, "InstructionSet")
        .value("NONE", dllm::simd::InstructionSet::NONE)
        .value("SSE42", dllm::simd::InstructionSet::SSE42)
        .value("AVX", dllm::simd::InstructionSet::AVX)
        .value("AVX2", dllm::simd::InstructionSet::AVX2)
        .value("AVX512", dllm::simd::InstructionSet::AVX512);
}
