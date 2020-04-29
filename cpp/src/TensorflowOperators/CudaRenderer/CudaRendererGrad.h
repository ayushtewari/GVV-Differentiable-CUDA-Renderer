//==============================================================================================//
// Classname:
//      CudaRendererGrad
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

#include "../../Renderer/CUDABasedRasterizationGrad.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class CudaRendererGrad : public OpKernel 
{
	//functions

	public:

		explicit CudaRendererGrad(OpKernelConstruction* context);
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

		CUDABasedRasterizationGrad* cudaBasedRasterizationGrad;

		//GPU input
		const float* d_inputRenderBufferGrad;
		const float* d_inputVertexPos;
		const float* d_inputVertexColor;
		const float* d_inputTexture;
		const float* d_inputSHCoeff;
		const float* d_inputVertexNormal;
		const float* d_inputBaryCentricBuffer;
		const int*   d_inputFaceBuffer;
		const float* d_inputTargetImage;
		const float* d_inputTargetImageGrad;

		//GPU output
		float*	d_outputVertexPosGrad;
		float*	d_outputVertexColorGrad;
		float*  d_outputTextureGrad;
		float*	d_outputSHCoeffGrad;

};

//==============================================================================================//

