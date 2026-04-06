cmake_minimum_required(VERSION 3.15)
project(HyperionEngineProject CXX)

set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT MSVC)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -O3)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find LibTorch
find_package(Torch REQUIRED)

set(HYPERION_SOURCES
    src/cpp/main.cpp
    src/cpp/core/bitboard.cpp
    src/cpp/core/movegen.cpp
    src/cpp/core/position.cpp
    src/cpp/core/zobrist.cpp
    src/cpp/search/search.cpp
    src/cpp/search/tt.cpp
    src/cpp/search/move_encoder.cpp
    src/cpp/nn_inference/nn_inference.cpp
)

add_executable(HyperionEngine ${HYPERION_SOURCES})

target_include_directories(HyperionEngine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${TORCH_INCLUDE_DIRS} 
)

if(NOT MSVC)
    target_compile_options(HyperionEngine PRIVATE
        -Wno-pedantic
        -Wno-unused-parameter
        -Wno-dangling-reference
    )
endif()

target_link_libraries(HyperionEngine PRIVATE ${TORCH_LIBRARIES})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET HyperionEngine POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${TORCH_DLLS}
      $<TARGET_FILE_DIR:HyperionEngine>)
endif()

option(HYPERION_ENABLE_BMI2 "Enable BMI2 optimizations" ON)
if(HYPERION_ENABLE_BMI2 AND NOT MSVC)
    target_compile_options(HyperionEngine PRIVATE -mbmi2)
endif()