# Minimum CMake version required
# this is here to make sure that all the commands used can be run by the user's CMake Version
cmake_minimum_required(VERSION 3.15)

# Project name and languages (CXX for C++)
# HyperionEngineProject is the name of the project, and CXX means we are using C++ for the project
project(HyperionEngineProject CXX)

# --- Standard C++ Settings ---
set(CMAKE_CXX_STANDARD 17) # tells the compiler to use C++17, can be changed to 20 too if you want
set(CMAKE_CXX_STANDARD_REQUIRED True) # prevents compilation of the compiler that doesn't support the above specified version
set(CMAKE_CXX_EXTENSIONS OFF) # no compiler specific extensions, just use the C++ standard

# --- CMake Option for BMI2 ---
option(HYPERION_ENABLE_BMI2 "Enable BMI2 instruction set optimizations (for PEXT/PDEP)" ON)

# --- Compiler Flags ---
# Add common warning flags
if(MSVC) # MSVC means the Miscrosoft Visual C++ compiler
    # Flags for Microsoft Visual C++
    # Consider /permissive- for stricter conformance if needed
    add_compile_options(/W4 /WX) # these are all flags that are added when compling the C++ code; W4 = high warning level, WX = treat warnings as errors
else()
    # Flags for GCC/Clang, the one we are likely using
    add_compile_options(-Wall -Wextra -Wpedantic -Werror) # -Wall: enables a good set of warnings | -Wextra: enables some more warnings not covered by -Wall | -Wpendantic: enables warnings for code that isn't strictly standard C++ compliant | -Werror: treats all warnings as errors
endif()

# --- Output Directories (Optional, for cleaner structure) ---
# These control where CMake puts the final compiled files (libraries and executables) within the build directory
# ${CMAKE_BINARY_DIR} is the current build directory (hyperion/build/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # CMAKE_ARCHIVE_OUTPUT_DIRECTORY is for static libaries (.a or .lib files)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # CMAKE_LIBRARY_OUTPUT_DIRECTORY is for shared libarie (.so or .dll)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # CMAKE_RUNTIME_OUTPUT_DIRECTORY is for the executable created (.exe)

# --- Engine Core Library ---
# makes everything in the src/cpp/core directory be able to be referenced as the "ENGINE_CORE_SOURCES" list
# ! IF ADDING NEW SOURCE (.cpp) FILES, MAKE SURE TO ADD THEM HERE ! (header files dont need to be added)
set(ENGINE_CORE_SOURCES
    src/cpp/core/bitboard.cpp
    src/cpp/core/movegen.cpp
    src/cpp/core/position.cpp
    src/cpp/core/zobrist.cpp
    # constants.hpp is header-only or included, not compiled directly unless it's a .cpp
)

# this makes everything in the ENGINE_CORE_SOURCES list into a static library called EngineCore
#   - a static library is a collection of object files that are linked into the executable at compile time
add_library(EngineCore STATIC ${ENGINE_CORE_SOURCES})


# this tells the compiler where to look for header files (.hpp) when compiling the EngineCore library and anything that uses EngineCore.
# the PUBLIC keyword means that files within EngineCore (ex: bitboard.cpp) can use these include paths, AND
# AND any other target that links against EngineCore (like EngineSearch or EngineUCI) will also inherently get these include paths 
# this means that files within EngineCore can include headers like `#include "core/bitboard.hpp"` or `#include "bitboard.hpp"` directly, AND
# AND files that use EngineCore can include headers like "core/bitboard.hpp" or "bitboard.hpp" as well.
target_include_directories(EngineCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core # For includes like "constants.hpp" from within core
)

# --- Engine Search Library ---
# this does the same thing as above, but for the search library (read lines 32-42)
set(ENGINE_SEARCH_SOURCES
    src/cpp/search/eval.cpp
    src/cpp/search/search.cpp
    src/cpp/search/tt.cpp
)

# same thing as above, but for the EngineSearch library (read lines 44-46)
add_library(EngineSearch STATIC ${ENGINE_SEARCH_SOURCES})

# same thing as above, but for the EngineSearch library (read lines 49-57)
target_include_directories(EngineSearch PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/search
)

if(HYPERION_ENABLE_BMI2)
    message(STATUS "BMI2 optimizations requested for EngineCore.")
    if(MSVC)
        # For MSVC, /arch:AVX2 enables AVX2 which implies BMI2.
        # This will lead to __AVX2__ being defined.
        target_compile_options(EngineCore PRIVATE /arch:AVX2)
        message(STATUS "  MSVC: Added /arch:AVX2 to EngineCore.")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # For GCC/Clang, use -mbmi2
        # This will lead to __BMI2__ being defined.
        include(CheckCXXCompilerFlag) # Include the module for checking flags
        check_cxx_compiler_flag("-mbmi2" COMPILER_SUPPORTS_BMI2)
        if(COMPILER_SUPPORTS_BMI2)
            target_compile_options(EngineCore PRIVATE -mbmi2)
            message(STATUS "  GCC/Clang: Added -mbmi2 to EngineCore.")
        else()
            message(WARNING "  GCC/Clang: Compiler does not support -mbmi2. BMI2 optimizations for EngineCore will not be enabled.")
        endif()
    else()
        message(WARNING "  Unsupported compiler for BMI2 flags: ${CMAKE_CXX_COMPILER_ID}. BMI2 optimizations for EngineCore may not be enabled.")
    endif()
else()
    message(STATUS "BMI2 optimizations are disabled for EngineCore.")
endif()

# this tells the compiler that the EngineSearch library needs code from the EngineCore library (linkning) in order to compile and link correctly
# the PRIVATE keywords mean that EngineSearch itself needs EngineCore, but other targets that link against EngineSearch don't directly ->
# -> need to know about EngineCore for linking puroposes, though they can still get EngineCore thought its PUBLIC directories.
# if things that need to use EngineSearch also need EngineCore for things outside of EngineSearch, then it could be PUBLIC, but since EngineCore ->
# -> is only used internally by EngineSearch, we can keep it PRIVATE.
target_link_libraries(EngineSearch PRIVATE EngineCore)

# --- Engine UCI Library ---
# same thing as above, but for UCI (read lines 32-42)
set(ENGINE_UCI_SOURCES
    src/cpp/uci/engine.cpp
    src/cpp/uci/uci.cpp
)

# same thing as above, but for the EngineUCI library (read lines 44-46)
add_library(EngineUCI STATIC ${ENGINE_UCI_SOURCES})

# same thing as above, but for the EngineUCI library (read lines 49-57)
target_include_directories(EngineUCI PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/uci
)

# since EngineUCI will use stuff from EngineSearch, we need to link it. EngineUCI won't have access to EngineCore, but it will have access to ->
# -> EngineSearch, which has access to EngineCore through its PUBLIC include directories.
target_link_libraries(EngineUCI PRIVATE EngineSearch)

# --- Main Hyperion Engine Executable ---
# this creates the main executable that will be run when using the engine. the main() function for this .exe uses the main() function in main.cpp
add_executable(HyperionEngine src/cpp/main.cpp)


# the Hyperion Engine executable will need include paths os its dependencies this adds /hyperion/src/cpp as an include path specifically for compiling HyperionEngine
target_include_directories(HyperionEngine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)

# HyperionEngine should not need to know about EngineCore or EngineSearch directly, but it will get them through EngineUCI.
# the code in main.cpp will only use EngineUCI, which uses the other libraries internally, but since those libraries were previously linked to EngineUCU ->
# -> they will automatically be linked to HyperionEngine as well.
target_link_libraries(HyperionEngine PRIVATE EngineUCI)

# --- TestBitboard Executable ---
# this will make a separate executable for testing the bitboard functionality, the same things from above (lines 103-114) apply here, but ->
# -> now we are creating a separate executable for testing purposes.
# test_bitboard.cpp is in core, so it will use EngineCore
add_executable(TestBitboard src/cpp/core/test_bitboard.cpp)
# see lines 108-109
target_include_directories(TestBitboard PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
# see lines 111-114
target_link_libraries(TestBitboard PRIVATE EngineCore)

add_executable(TestMovegen src/cpp/core/perft.cpp)
target_include_directories(TestMovegen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
target_link_libraries(TestMovegen PRIVATE EngineCore)
target_compile_options(TestMovegen PRIVATE -Ofast)

add_executable(TestPuzzles src/cpp/core/test_puzzle.cpp)

target_include_directories(TestPuzzles PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
target_link_libraries(TestPuzzles PRIVATE EngineSearch)
target_compile_options(TestPuzzles PRIVATE -O3)

# --- Future NN Library Integration (Placeholder) ---
# When you integrate ONNX Runtime or another NN library:
# 1. Find the package:
#    find_package(ONNXRuntime REQUIRED) # You might need to set ONNXRUNTIME_DIR
# 2. Link the main engine (or relevant library) against it:
#    target_link_libraries(HyperionEngine PRIVATE ONNXRuntime::onnxruntime)
#    (The exact target name ONNXRuntime::onnxruntime might vary based on how find_package works for it)
# 3. Add its include directories if not handled by find_package's imported targets:
#    target_include_directories(HyperionEngine PRIVATE ${ONNXRuntime_INCLUDE_DIRS})

# --- Output ---
message(STATUS "Configuring HyperionEngineProject")
message(STATUS "  Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  Build directory:  ${CMAKE_BINARY_DIR}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}") # Will be empty if not set via -DCMAKE_BUILD_TYPE
if(HYPERION_ENABLE_BMI2)
    message(STATUS "  BMI2 Optimizations: ENABLED (if supported by compiler)")
else()
    message(STATUS "  BMI2 Optimizations: DISABLED")
endif()

# HOW TO BUILD:
# 1) open a terminal in the hyperion directory
# 2) cd into the build directory (hyperion/build/)
# 3) run `cmake -G "MinGW Makefiles" ..`
#       - explanation: `-G "MinGW Makefiles"` tells CMake to use the MinGW Makefiles generator, which is needed for compiling with MinGW on Windows.
#                       and `..` tells CMake to look for the CMakeLists.txt file in the above/parent directory (hyperion/).
# 4) run `mingw32-make` (still in the build directory)
#       - if this doesn't work, try just `make` instead   
# 5) to run the executables, do `.\bin\NAME_OF_EXECUTABLE.exe` (e.g., `.\bin\HyperionEngine.exe` or `.\bin\TestBitboard.exe`)

# NOTE: If you are just making changes to the source code and not the CMakeLists.txt, you can just do the following:
# 1) cd into the build directory (hyperion/build/)
# 2) run `mingw32-make` (or `make` if that doesn't work)
# 3) run the executables as described above

# HOW TO BUILD TO RUN FAST:
# 1) cmake -G "MinGW Makefiles" -DHYPERION_ENABLE_BMI2=ON -DCMAKE_BUILD_TYPE=Release ..
#       - explanation '-G "MinGW Makefiles" tells CMake to generate Makefiles for MinGW
#       and -DYPERION_ENABLE_BMI2=0 explicitly enables BMI2 option
#       and -DCMAKE_BUILD_TYPE=Release tells cmake to use optimizations like -O3
# 2) run 'mingw32-make -jN # replace N with the number of cores for faster compilation
# 3) to run the executables, do `.\bin\NAME_OF_EXECUTABLE.exe` (e.g., `.\bin\HyperionEngine.exe` or `.\bin\TestBitboard.exe`)
