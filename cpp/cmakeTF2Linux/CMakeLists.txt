#############################################################################################
#cmake version

CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
MESSAGE("++++ Required CMake Version: 3.9")

#############################################################################################
#TO BE CUSTOMIZED

SET (CUDA_PATH /usr/lib/cuda-10.1/bin/nvcc)
SET (TENSORFLOW_PATH /HPS/RTMPC/work/Programs/Anaconda3-4.1.1/envs/tf201/lib/python3.5/site-packages/tensorflow_core)
SET (TFVERSION "2.1.0")

#############################################################################################
#Project with CUDA

SET(PROJECT_NAME CustomTensorFlowOperators)

SET( CUDACXX ${CUDA_PATH})
SET( CMAKE_CUDA_COMPILER ${CUDA_PATH})

PROJECT(${PROJECT_NAME} CXX CUDA)
MESSAGE("++++ Project name: "${PROJECT_NAME})

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

MESSAGE("++++ Project directory: "${projectDir})

#############################################################################################
#DEPENDENCIES

MESSAGE("++++ Prepare includes and libs")

#-------INCLUDES-------
	
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../thirdParty/Shared/cutil/inc/)

#-------TENSORFLOW-------

SET (tensorflowLib ${TENSORFLOW_PATH}/python/_pywrap_tensorflow_internal.so)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/)
MESSAGE("++++ Tensorflow directory: "${TENSORFLOW_PATH})

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

SET(CUDA_NVCC_FLAGS "-O3" "-arch=sm_70" "--default-stream per-thread" "-Xptxas -v" CACHE STRING "nvcc flags" FORCE)
SET(CMAKE_CUDA_FLAGS "-arch=sm_70")

##############################################################################################
#OUTPUT LIB

MESSAGE("++++ Set output lib properties")

ADD_DEFINITIONS(-DGOOGLE_CUDA -D_USE_MATH_DEFINES)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${projectDir}/binaries/Linux/Release")
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${projectDir}/binaries/Linux/Release")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${projectDir}/binaries/Linux/Release")

IF (TFVERSION EQUAL  "2.0.0-beta0" OR TFVERSION EQUAL  "2.1.0" )
	MESSAGE("TF VERSION IS 2.0.0-beta0 or 2.1.0  --> SET -D_GLIBCXX_USE_CXX11_ABI=0")
	SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
ELSE()
	MESSAGE("TF VERSION IS DIFFERENT FROM 2.0.0-beta0 or 2.1.0  --> SET -D_GLIBCXX_USE_CXX11_ABI=1")
	SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
ENDIF()

ADD_LIBRARY(${PROJECT_NAME} SHARED ${cppFiles} ${cudaFiles})

##############################################################################################
#OUTPUT LIB

MESSAGE("++++ Specifiy the output")

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${tensorflowLib})




