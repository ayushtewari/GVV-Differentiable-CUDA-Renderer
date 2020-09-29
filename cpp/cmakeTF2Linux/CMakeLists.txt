#############################################################################################
#cmake version

CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
MESSAGE("++++ Required CMake Version: 3.9")

#############################################################################################
#TO BE CUSTOMIZED

#TF 210
#SET (CUDA_PATH /usr/lib/cuda-10.1/bin/nvcc)
#SET (TENSORFLOW_PATH /HPS/RTMPC/work/Programs/Anaconda3-4.1.1/envs/tf210/lib/python3.5/site-packages/tensorflow_core)
#SET (BINARY_DIR /HPS/RTMPC/work/CudaRenderer/cpp/binaries/Linux/ReleaseTF2GCC8)
#SET (TFVERSION "2.1.0")

#TF 220 gpu20
#SET (CUDA_PATH /usr/lib/cuda-10.1/bin/nvcc)
#SET (TENSORFLOW_PATH /HPS/RTMPC/work/Programs/Anaconda3-4.1.1/envs/tf220/lib/python3.5/site-packages/tensorflow)
#SET (BINARY_DIR /HPS/RTMPC/work/CudaRenderer/cpp/binaries/Linux/ReleaseTF22GCC8)
#SET (TFVERSION "2.2.0")
#SET (CUDA_ARCH "-arch=sm_75")

#TF 220 d2volta recon
SET (CUDA_PATH /usr/lib/cuda-10.1/bin/nvcc)
SET (TENSORFLOW_PATH /HPS/RTMPC/work/Programs/Anaconda3-4.1.1/envs/tf220/lib/python3.5/site-packages/tensorflow)
SET (BINARY_DIR /HPS/RTMPC/work/CudaRenderer/cpp/binaries/Linux/Release)
SET (TFVERSION "2.2.0")
SET (CUDA_ARCH "-arch=sm_70")

# GARCHING
#SET (CUDA_PATH    /mpcdf/soft/SLE_12/packages/x86_64/cuda/10.1.243/bin/nvcc)
#SET (CUDA_INCLUDE /mpcdf/soft/SLE_12/packages/x86_64/cuda/10.1.243/include)
#SET (TENSORFLOW_PATH /u/mhaberma/.local/lib/python3.7/site-packages/tensorflow_core)
#SET (BINARY_DIR  /u/mhaberma/RTMPC/work/CudaRenderer/cpp/binaries/Linux/Release)
#SET (TFVERSION "2.1.0")

#############################################################################################
#Project with CUDA

SET(PROJECT_NAME CustomTensorFlowOperators)

SET( CUDACXX ${CUDA_PATH})
SET( CMAKE_CUDA_COMPILER ${CUDA_PATH})

PROJECT(${PROJECT_NAME} CXX CUDA)
MESSAGE("++++ Project name: " ${PROJECT_NAME})

#############################################################################################
#Misc

SET(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

SET( CMAKE_EXE_LINKER_FLAGS  "-ltensorflow_framework" )
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET (BUILD_SHARED_LIBS TRUE)

SET(projectDir "")
STRING(REPLACE "/cmakeTF2Linux"
       "" projectDir
       ${CMAKE_SOURCE_DIR})

MESSAGE("++++ Project directory: " ${projectDir})

#############################################################################################
#DEPENDENCIES

MESSAGE("++++ Prepare includes and libs")

#-------INCLUDES-------
	
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../thirdParty/Shared/cutil/inc/)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE})

#-------TENSORFLOW-------

SET (tensorflowLib ${TENSORFLOW_PATH}/python/_pywrap_tensorflow_internal.so)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/)
MESSAGE("++++ Tensorflow directory: " ${TENSORFLOW_PATH})

##############################################################################################
#ADD FILES

MESSAGE("++++ Add source files")

FILE(GLOB_RECURSE cppFiles 			
	${CMAKE_SOURCE_DIR}/../src/Utils/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Utils/*.h

	${CMAKE_SOURCE_DIR}/../src/TensorflowOperators/CudaRenderer/*.cpp
	${CMAKE_SOURCE_DIR}/../src/TensorflowOperators/CudaRenderer/*.h

	${CMAKE_SOURCE_DIR}/../src/Renderer/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Renderer/*.h
)

FILE(GLOB cudaFiles
	${CMAKE_SOURCE_DIR}/../src/TensorflowOperators/CudaRenderer/*.cu
	${CMAKE_SOURCE_DIR}/../src/Renderer/*.cu
)

##############################################################################################
#CUDA COMPILATION FLAGS

MESSAGE("++++ Set CUDA compilation properties")

SET(CUDA_NVCC_FLAGS "-O3" ${CUDA_ARCH} "--default-stream per-thread" "-Xptxas -v" CACHE STRING "nvcc flags" FORCE)
SET(CMAKE_CUDA_FLAGS ${CUDA_ARCH})

##############################################################################################
#OUTPUT LIB

MESSAGE("++++ Set output lib properties")

ADD_DEFINITIONS(-DGOOGLE_CUDA -D_USE_MATH_DEFINES)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BINARY_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})

IF (TFVERSION EQUAL  "2.0.0-beta0" OR TFVERSION EQUAL  "2.1.0" OR TFVERSION EQUAL  "2.2.0")
	MESSAGE("TF VERSION IS 2.0.0-beta0 or 2.1.0  --> SET -D_GLIBCXX_USE_CXX11_ABI=0")
	SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -O3 ")
	MESSAGE("CMake CXX Flags: ${CMAKE_CXX_FLAGS}")
ELSE()
	MESSAGE("TF VERSION IS DIFFERENT FROM 2.0.0-beta0 or 2.1.0  --> SET -D_GLIBCXX_USE_CXX11_ABI=1")
	SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -O3 ")
	MESSAGE("CMake CXX Flags: ${CMAKE_CXX_FLAGS}")
ENDIF()

SET( CMAKE_BUILD_TYPE "Release" FORCE)

ADD_LIBRARY(${PROJECT_NAME} SHARED ${cppFiles} ${cudaFiles})

##############################################################################################
#OUTPUT LIB

MESSAGE("++++ Specifiy the output")
MESSAGE("Build type: ${CMAKE_BUILD_TYPE}")

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${tensorflowLib})




