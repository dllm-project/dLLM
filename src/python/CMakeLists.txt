# Python bindings build configuration

find_package(pybind11 REQUIRED)

pybind11_add_module(dllm_cpp MODULE
    python_bridge.cpp
)

target_include_directories(dllm_cpp PRIVATE ${CMAKE_SOURCE_DIR}/src/cpp)
target_link_libraries(dllm_cpp PRIVATE dllm_tensor dllm_engine dllm_comm)

set_target_properties(dllm_cpp PROPERTIES
    OUTPUT_NAME "dllm"
    PREFIX ""
)

install(TARGETS dllm_cpp LIBRARY DESTINATION .)