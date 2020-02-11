//==============================================================================================//
// Classname:
//      CudaRenderer
//
//==============================================================================================//
// Description:
//      Todo
//
//==============================================================================================//
// Input:
//		Todo
//		
//
//==============================================================================================//
// Output:
//		Todo
//
//==============================================================================================//

#define NOMINMAX

//==============================================================================================//

#pragma once

//==============================================================================================//

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../../Renderer/Mesh/CUDABasedRasterization.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class CudaRenderer : public OpKernel 
{
	//functions

	public:

		explicit CudaRenderer(OpKernelConstruction* context);
		void Compute(OpKernelContext* context);
	
	private:
		
		void setupInputOutputTensorPointers(OpKernelContext* context);

	//variables

	public:

	private:

		//operator settings and flags
		std::string cameraFilePath;
		std::string meshFilePath;

		int numberOfBatches;
		int numberOfCameras;
		int numberOfPoints;
		int renderResolutionU;
		int renderResolutionV;

		CUDABasedRasterization* cudaBasedRasterization;
		camera_container* cameras;
		trimesh* mesh;

		//GPU input
		const float* d_inputDataPointerPointsGlobalSpace;

		float*	d_outputBarycentricCoordinatesBuffer;
		int*	d_outputFaceIDBuffer;
		int*	d_outputDepthBuffer;
		float*	d_outputRenderBuffer;
		float*	d_outputVertexColorBuffer;
};

//==============================================================================================//

