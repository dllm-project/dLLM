# Python bindings build configuration

find_package(pybind11 REQUIRED)

pybind11_add_module(dllm_cpp MODULE
    python_bridge.cpp
)

target_include_directories(dllm_cpp PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/tensor
    ${CMAKE_SOURCE_DIR}/src/cpp/engine
    ${CMAKE_SOURCE_DIR}/src/cpp/comm
)
target_link_libraries(dllm_cpp PRIVATE dllm_tensor dllm_engine dllm_comm)

set_target_properties(dllm_cpp PROPERTIES
    OUTPUT_NAME "dllm"
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

install(TARGETS dllm_cpp LIBRARY DESTINATION .)

# Setup.py configuration for wheel building
configure_file(
    ${CMAKE_SOURCE_DIR}/src/python/setup.py.in
    ${CMAKE_BINARY_DIR}/setup.py
    @ONLY
)