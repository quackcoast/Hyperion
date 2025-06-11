# Hyperion Chess Engine
Hyperion is a MCTS-NN based chess engine built primarily in C++. The Engine is still in development, but should be done by the end of Summer 2025.

## Perft test results:
- Nodes per second: 4.24985e+07 (42.5 Million NPS)

## How to build?
1. Open a terminal in the `hyperion` directory

2. Next, cd into the build directory (`hyperion/build/`)

3) Now run `cmake -G "MinGW Makefiles" ..`

4) Still in the build directory, run `mingw32-make` (if this doesn't work, try just `make` instead)
5) Run the executables by running the command `.\bin\HyperionEngine.exe` 

## Tasks Completed
- [x] Chess Logic Library
- [ ] Basic Monte-Carlo Tree Search Implementation
- [ ] Initial Neural Network Creation (Supervised Learning)
- [ ] Initial Neural Network Implementation 
- [ ] Neural Network Self-Play (Reinforcement Learning)
- [ ] Refinements/Advanced Features
