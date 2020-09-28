#############################################################################################
#cmake version

CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
MESSAGE("++++ Required CMake Version: 3.9")

#############################################################################################
#TO BE CUSTOMIZED 

#Tensorflow 2.0.0-beta0

SET (CUDADIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/")
SET (TENSORFLOW_PATH D:/Anaconda/Anaconda3/tf2/Lib/site-packages/tensorflow)

#Fix change it manually to cuda 10.0

#############################################################################################
#CUDA 

SET( CUDACXX ${CUDADIR})
SET( CMAKE_CUDA_COMPILER ${CUDADIR})

#############################################################################################
#Project 

SET(PROJECT_NAME CustomTensorFlowOperators)
PROJECT(${PROJECT_NAME} CXX CUDA)
MESSAGE("++++ Project name: "${PROJECT_NAME})

#############################################################################################
#Misc

SET(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
SET( CMAKE_EXE_LINKER_FLAGS  "-ltensorflow_framework" )
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET (BUILD_SHARED_LIBS TRUE)

SET(projectDir "")
STRING(REPLACE "/cmakeTF2Windows"
       "" projectDir
       ${CMAKE_SOURCE_DIR})

MESSAGE("++++ Project directory: "${projectDir})

#############################################################################################
#others

MESSAGE("++++ Prepare includes and libs")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../thirdParty/Shared/cutil/inc/)

#############################################################################################
#tensorflow

MESSAGE("++++ Tensorflow lib")
SET (tensorflowLib ${TENSORFLOW_PATH}/python/_pywrap_tensorflow_internal.lib)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/external/eigen_archive/)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/external/com_google_absl/)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/external/protobuf_archive/src/)

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
#CUDA COMPILATION

MESSAGE("++++ Set CUDA compilation properties")

SET(CUDA_NVCC_FLAGS "-O3" "-arch=sm_61" "--default-stream per-thread" "-Xptxas -v" CACHE STRING "nvcc flags" FORCE)
SET(CMAKE_CUDA_FLAGS "-arch=sm_61")

##############################################################################################
#OUTPUT LIB

MESSAGE("++++ Set output lib properties")

ADD_DEFINITIONS(-DWIN32 -D_WINDOWS -DNDEBUG -DGOOGLE_CUDA -D_USE_MATH_DEFINES)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${projectDir}/binaries/Win64")
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${projectDir}/binaries/Win64")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${projectDir}/binaries/Win64")

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

ADD_LIBRARY(${PROJECT_NAME} SHARED ${cppFiles} ${cudaFiles})

##############################################################################################
#ADD STRUCTURE IN SOLUTION

MESSAGE("++++ Add structure to the visual studio solution")
FILE( GLOB_RECURSE SRCS REGULAR_EXPRESSION "${CMAKE_SOURCE_DIR}/../src/*")


FOREACH(FILE ${SRCS}) 
    # Get the directory of the source file
    GET_FILENAME_COMPONENT(PARENT_DIR "${FILE}" DIRECTORY)

    # Remove common directory prefix to make the group
    STRING(REPLACE "${CMAKE_SOURCE_DIR}/../src/" "" GROUP "${PARENT_DIR}")

    # Make sure we are using windows slashes
    STRING(REPLACE "/" "\\" GROUP "${GROUP}")

    SOURCE_GROUP("${GROUP}" FILES "${FILE}")
endforeach()

##############################################################################################
#OUTPUT LIB

MESSAGE("++++ Specifiy the output")

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${tensorflowLib} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${tensorflowLib})




