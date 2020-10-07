# GVV-Differentiable-CUDA-Renderer
<img src="logo/teaserRender.png" border="0" width="150" class="center">

This is a simple and efficient differentiable rasterization-based renderer which has been used in several [GVV publications](https://gvv.mpi-inf.mpg.de/GVV_Projects.html). The implementation is free of most third-party libraries such as OpenGL. The core implementation is in CUDA and C++. We use the layer as a custom Tensorflow op.  

# Features 
The renderer supports the following features:
- Shading based on spherical harmonics illumination. This shading model is differentiable with respect to geometry, texture, and lighting. 
- Different visualizations, such as normals, UV coordinates, phong-shaded surface, spherical-harmonics shading and colors without shading. 
- Texture map lookups.
- Rendering from multiple camera views in a single batch

Visibility is not differentiable. We also do not approximate the gradients due to occlusions. This simple strategy works for many use cases such as fitting parametric shape models to images. 

### Requirements (tested versions):
- Tensorflow 2.2.0
- GPU that supports compute capability 7.0
- CUDA 10.0
- CUDNN 7.6.1
- Python 3.5
- CMake 3.9 and higher

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

### Citations
Please cite the following papers if you use the renderer in your project:

    @inproceedings{deepcap,
    title = {DeepCap: Monocular Human Performance Capture Using Weak Supervision},
    author = {Habermann, Marc and Xu, Weipeng and Zollhoefer, Michael and Pons-Moll, Gerard and Theobalt, Christian},
    booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {jun},
    organization = {{IEEE}},
    year = {2020},
    }	
    
    @misc{r2020learning,
    title={Learning Complete 3D Morphable Face Models from Images and Videos}, 
    author={Mallikarjun B R and Ayush Tewari and Hans-Peter Seidel and Mohamed Elgharib and Christian Theobalt},
    year={2020},
    eprint={2010.01679},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }


### License 
The contents of this repository, and the pretrained models are made available under CC BY 4.0. Please read the [license terms](https://creativecommons.org/licenses/by/4.0/legalcode).

### Contributors
- [Marc Habermann](https://people.mpi-inf.mpg.de/~mhaberma/)
- [Mallikarjun B R](https://people.mpi-inf.mpg.de/~mbr/)
- Linjie Liu
- [Ayush Tewari](https://people.mpi-inf.mpg.de/~atewari/)
