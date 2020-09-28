# GVV-Differentiable-Renderer

This is a simple differentiable rasterization-based renderer which has been used in several GVV publications. The implemtation is free of most third-party libraries such as OpenGL. Rasterization is implemented in CUDA. 

# Features 
The renderer supports the following features:
1. Shading based on spherical harmonics illumination. This shading model is differentiable. 
2. Different visualizations, such as normals, UV coordinates, phong-shaded surface, spherical-harmonics shading and surface color without shading. 
3. Texture map lookups.


### Requirements (supported versions we tested):
1. Tensorflow 2.0.0-beta0
2. GPU that supports compute capability 7.0
3. CUDA 10.0
4. CUDNN 7.6.1
5. Python 3.5
6. CMake 3.9 and higher

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
1. Test for forward pass in python/test_render.py
2. Gradient tests in python/test_gradients*.py