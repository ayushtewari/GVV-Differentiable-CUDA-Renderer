# GVV-Differentiable-Renderer

This is a simple differentiable rasterization-based renderer which has been used in several GVV publications. The implementation is free of most third-party libraries such as OpenGL. Rasterization is implemented in CUDA. 

# Features 
The renderer supports the following features:
. Shading based on spherical harmonics illumination. This shading model is differentiable. 
. Different visualizations, such as normals, UV coordinates, phong-shaded surface, spherical-harmonics shading and colors without shading. 
. Texture map lookups.
. Rendering from multiple camera views in a single batch

### Requirements (tested versions):
. Tensorflow 2.0.0-beta0
. GPU that supports compute capability 7.0
. CUDA 10.0
. CUDNN 7.6.1
. Python 3.5
. CMake 3.9 and higher

### Installation Linux:
1. Clone the repository 
2. Change the cmake in cpp/cmakeTF2Linux (lines under "TO BE CUSTOMIZED") pointing to your CUDA and TF path
3. Run createBuildLinux.sh
4. Navigate to cpp/build/Linux and run make 
5. If compilation finished, there should be the files cpp/binaries/Linux/Release/libCustomTensorFlowOperators.so

### Installation Windows:
1. Clone the repository 
2. Change the cmake in cpp/cmakeTF2Windows (lines under "TO BE CUSTOMIZED") pointing to your CUDA and TF path
3. Run createBuildWin64.bat
4. Navigate to cpp/build/Win64 and compile with Visual studio (tested on VS 2015 x64)
5. If compilation finished, there should be the files cpp/binaries/Win64/Release/CustomTensorFlowOperators.dll

### Examples
1. Forward pass in python/test_render.py
2. Gradient tests in python/test_gradients*.py

### Contributors

