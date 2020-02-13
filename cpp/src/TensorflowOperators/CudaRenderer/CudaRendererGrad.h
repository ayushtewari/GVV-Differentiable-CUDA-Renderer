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

		//GPU input
		const float* d_inputVertexColorBufferGrad;
		const float* d_inputVertexPos;
		const float* d_inputVertexColor;
		const float* d_inputTexture;
		const float* d_inputSHCoeff;
		const float* d_vertexNormal;
		const float* d_baryCentricBuffer;
		const float* d_faceBuffer;
		const float* d_vertexColorBuffer;

		//GPU output
		float*	d_outputVertexPosGrad;
		float*	d_outputVertexColorGrad;
		float*	d_outputSHCoeffGrad;

};

//==============================================================================================//

