//==============================================================================================//
// Classname:
//      CudaRenderer
//
//==============================================================================================//
// Description:
//      Implements a cuda based rasterizer that is differentiable
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

#include "../../Renderer/CUDABasedRasterization.h"

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

		int numberOfBatches;
		int numberOfCameras;
		int numberOfPoints;
		int renderResolutionU;
		int renderResolutionV;
		int textureResolutionU;
		int textureResolutionV;

		std::string albedoMode;
		std::string shadingMode;

		CUDABasedRasterization* cudaBasedRasterization;

		//GPU input
		const float* d_inputVertexPos;
		const float* d_inputVertexColor;
		const float* d_inputTexture;
		const float* d_inputSHCoeff;
		const float* d_inputTargetImage;

		//GPU output
		float*	d_outputBarycentricCoordinatesBuffer;
		int*	d_outputFaceIDBuffer;
		float*	d_outputRenderBuffer;

		float*	d_outputVertexNormal;
		float*	d_outputTargetImage;
		float*	d_outputNormalMap;
};

//==============================================================================================//

