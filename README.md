# Hyperion Chess Engine
Hyperion is a MCTS-NN based chess engine built primarily in C++. The Engine is still in development with plans to add reinforcement learning soon!

## Project Structure (Two Builds)
There are two different and separate builds for this project:

1. **Python (Training)**
   - The Python code (using PyTorch) is strictly used for training to create the neural network models.
   - To set up the Python environment, install the requirements by running:
     ```bash
     pip install -e .
     ```
   - Navigate to the training directory and run:  
     ```bash
     python train.py
     ```
2. **C++ (Engine and Inference)**
   - The C++ code is the actual chess engine. It uses the neural network files trained by the Python component (saved in `data/completed_models`) for its inferences during searches.

## Perft test results:
- Nodes per second: 4.39454e+07 (43.9 Million NPS)

## Prerequisites: LibTorch (For C++ Engine)
Because this engine uses a Neural Network, it requires **LibTorch** (the C++ API for PyTorch) to build.
1. Download LibTorch from the [PyTorch website](https://pytorch.org/).
   * **Linux:** Make sure to select the **cxx11 ABI** version.
   * **Windows:** Download the Release version. **IMPORTANT:** Do *not* use MinGW on Windows. You should instead use **MSVC**.
2. Extract the downloaded folder somewhere on your computer. You will need the path to this folder for the build steps below.

## How to build C++ Engine?
**CRITICAL:** It is super important that you build the project in **Release** mode, not Debug! The Debug mode does not work with LibTorch and will cause errors.

1. Open a terminal in the `hyperion` directory.

2. Next, create and `cd` into the build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Configure the project with CMake, making sure to provide the path to your extracted LibTorch folder and specifying the Release build type:
   ```bash
   cmake -DCMAKE_PREFIX_PATH="C:/absolute/path/to/your/libtorch" -DCMAKE_BUILD_TYPE=Release ..
   ```

4. Still in the build directory, compile the engine (Release config is passed to cmake build for MSVC):
   ```bash
   cmake --build . --config Release
   ```

5. Run the executable:
   ```bash
   .\bin\Release\HyperionEngine.exe
   ```
   *(Note: The exact path might be `.\bin\HyperionEngine.exe` on non-MSVC builds, such as Linux.)*

## How to build C++ Engine with performance enhancements:
1. Open a terminal in the `hyperion` directory.

2. Next, `cd` into the build directory (`hyperion/build/`).

3. Run CMake with optimization flags, the LibTorch path, and Release mode:
   ```bash
   cmake -DCMAKE_PREFIX_PATH="C:/absolute/path/to/your/libtorch" -DHYPERION_ENABLE_BMI2=ON -DCMAKE_BUILD_TYPE=Release ..
   ```

4. Still in the build directory, build using multiple cores for a faster compile time:
   ```bash
   cmake --build . --config Release -j 8
   ```

5. Run the executable:
   ```bash
   .\bin\Release\HyperionEngine.exe
   ```

## Tasks Completed
- [x] Chess Logic Library
- [x] Basic Monte-Carlo Tree Search Implementation
- [x] Initial Neural Network Creation (Supervised Learning)
- [x] Initial Neural Network Implementation 
- [ ] Neural Network Self-Play (Reinforcement Learning)
- [ ] Refinements/Advanced Features
