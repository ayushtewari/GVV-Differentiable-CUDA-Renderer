# Cuda Renderer 
Check our [project page](http://gvv.mpi-inf.mpg.de/) for additional information.

Description here!

#### NEW: Tensorflow implementation available!

### Requirements:
1. Tensorflow 2.0.0-beta0
2. GPU that supports compute capability 7.0
3. CUDA 10.0
4. CUDNN 7.6.1
5. Python 3.5
6. CMake 3.9 and higher

### Installation:
1. Clone the repository 
2. Change the cmake in cpp/cmakeTF2Linux / cpp/cmakeTF2Windows (lines under "TO BE CUSTOMIZED") pointing to your CUDA and TF path
3. Run createBuildLinux.sh/createBuildWin64.bat
4. Navigate to cpp/build/Linux and run make / Navigate to cpp/build/Win64 and compile with Visual studio (tested on VS 2015 x64)
5. If compilation finished, there should be the files cpp/binaries/Linux/Release/libCustomTensorFlowOperators.so / cpp/binaries/Win64/Release/CustomTensorFlowOperators.dll

### Examples
1. Run the simple test in python/test_render.py

### Citation:
	@Inproceedings{Cae+17,
	}
	
If you encounter any problems with the code, want to report bugs, etc. please contact us at mhaberma@mpi-inf.mpg.de.
