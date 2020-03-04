
//==============================================================================================//

#include <cuda_runtime.h> 
#include "../Utils/cudaUtil.h"
#include "../Utils/cuda_SimpleMatrixUtil.h"
#include "../Utils/RendererUtil.h"
#include "CUDABasedRasterizationGradInput.h"
#include "../Utils/CameraUtil.h"
#include "../Utils/IndexHelper.h"

//==============================================================================================//

// Cg - vertex color grad buffer
// Co - vertex color buffer 
// Al - albedo color buffer 
// Li - light component buffer
// Vc - vertex color (vertex space)
// Bc - bary centric co-ordinates
// No - normal buffer
// Nv - vertex normal 
// Nf - face normal 
// Vp - vertex position 
// Gm - sh coefficients


// vertex_color_grad = ( Cg * JCgAl * JAlVc )


/*
Get gradients for vertex color buffer
*/
__global__ void renderBuffersGradDevice(CUDABasedRasterizationGradInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input.numberOfCameras * input.w * input.h)
	{
		////////////////////////////////////////////////////////////////////////
		//INDEXING
		////////////////////////////////////////////////////////////////////////

		int3 index = index1DTo3D(input.numberOfCameras, input.h, input.w, idx);
		int idc = index.x;
		int idh = index.y;
		int idw = index.z;
		int4 idf = input.d_faceIDBuffer[idx];

		if (idf.w == -1)
			return;

		////////////////////////////////////////////////////////////////////////
		//INIT
		////////////////////////////////////////////////////////////////////////

		float3 bcc				= input.d_barycentricCoordinatesBuffer[idx];
		int3   faceVerticesIds  = input.d_facesVertex[idf.x];
		const float* shCoeff	= input.d_shCoeff + idc * 27;

		float3 vertexPos0 = input.d_vertices[faceVerticesIds.x];
		float3 vertexPos1 = input.d_vertices[faceVerticesIds.y];
		float3 vertexPos2 = input.d_vertices[faceVerticesIds.z];
		float3 vertexCol0 = input.d_vertexColor[faceVerticesIds.x];
		float3 vertexCol1 = input.d_vertexColor[faceVerticesIds.y];
		float3 vertexCol2 = input.d_vertexColor[faceVerticesIds.z];
		float3 vertexNor0 = input.d_vertexNormal[idc*input.N + faceVerticesIds.x];
		float3 vertexNor1 = input.d_vertexNormal[idc*input.N + faceVerticesIds.y];
		float3 vertexNor2 = input.d_vertexNormal[idc*input.N + faceVerticesIds.z];

		float3 pixAlb		= bcc.x * vertexCol0 + bcc.y * vertexCol1 + bcc.z * vertexCol2;
		float3 pixNormUn	= bcc.x * vertexNor0 + bcc.y * vertexNor1 + bcc.z * vertexNor2;
		float  pixNormVal	= sqrtf(pixNormUn.x*pixNormUn.x + pixNormUn.y*pixNormUn.y + pixNormUn.z*pixNormUn.z);
		float3 pixNorm = pixNormUn / pixNormVal;


		mat1x3 GVCB;
		GVCB(0, 0) = input.d_vertexColorBufferGrad[idx].x; 
		GVCB(0, 1) = input.d_vertexColorBufferGrad[idx].y; 
		GVCB(0, 2) = input.d_vertexColorBufferGrad[idx].z;

		////////////////////////////////////////////////////////////////////////
		//VERTEX COLOR GRAD
		////////////////////////////////////////////////////////////////////////

		float3 pixLight = getIllum(pixNorm, shCoeff);

		mat3x3 JCoAl;
		getJCoAl(JCoAl, pixLight);

		mat3x9 JAlVc;
		getJAlVc(JAlVc, bcc);

		mat1x9 gradVerCol = GVCB * JCoAl * JAlVc;

		addGradients9I(gradVerCol, input.d_vertexColorGrad, faceVerticesIds);

		////////////////////////////////////////////////////////////////////////
		//LIGHTING GRAD
		////////////////////////////////////////////////////////////////////////

		// jacobians
		mat3x3 JCoLi;
		getJCoLi(JCoLi, pixAlb);

		mat3x9 JLiGmR;
		getJLiGm(JLiGmR, 0, pixNorm);
		mat3x9 JLiGmG;
		getJLiGm(JLiGmG, 1, pixNorm);
		mat3x9 JLiGmB;
		getJLiGm(JLiGmB, 2, pixNorm);

		mat1x9 gradSHCoeffR = GVCB * JCoLi * JLiGmR;
		mat1x9 gradSHCoeffG = GVCB * JCoLi * JLiGmG;
		mat1x9 gradSHCoeffB = GVCB * JCoLi * JLiGmB;

		addGradients9(gradSHCoeffR, &input.d_shCoeffGrad[idc * 27]);
		addGradients9(gradSHCoeffG, &input.d_shCoeffGrad[idc * 27+9]);
		addGradients9(gradSHCoeffB, &input.d_shCoeffGrad[idc * 27+18]);

		////////////////////////////////////////////////////////////////////////
		//VERTEX POS GRAD
		////////////////////////////////////////////////////////////////////////

		mat3x3 JNoNu;
		getJNoNu(JNoNu, pixNormUn, pixNormVal);

		mat3x3 JLiNo;
		getJLiNo(JLiNo, pixNorm, (float *)shCoeff);

		mat3x3 TR;
		TR = getRotationMatrix(&input.d_cameraExtrinsics[3 * idc]);
		mat3x3 J;
		int idv;
		mat3x3 JNuNvx;
		JNuNvx.setIdentity();
		for (int i = 0; i < 3; i++)
		{
			//
			if (i == 0) { idv = faceVerticesIds.x; JNuNvx = bcc.x * JNuNvx; }
			else if (i == 1) { idv = faceVerticesIds.y; JNuNvx = bcc.y * JNuNvx; }
			else { idv = faceVerticesIds.z; JNuNvx = bcc.z * JNuNvx; }

			//
			int2 verFaceId = input.d_vertexFacesId[idv];

			//
			mat3x1 vi, vj, vk;
			for (int j = verFaceId.x; j < verFaceId.x + verFaceId.y; j++)
			{
				int faceId = input.d_vertexFaces[j];
				int3 v_index_inner = input.d_facesVertex[faceId];
				vi = TR * (mat3x1)input.d_vertices[v_index_inner.x];
				vj = TR * (mat3x1)input.d_vertices[v_index_inner.y];
				vk = TR * (mat3x1)input.d_vertices[v_index_inner.z];

				getJ(J, TR, vj, vi);

				// gradients
				mat1x3 gradVj = GVCB * JCoLi * JLiNo * JNoNu * JNuNvx * J;
				addGradients(gradVj, &input.d_vertexPosGrad[v_index_inner.y].x);
				mat1x3 gradVi = -GVCB * JCoLi * JLiNo * JNoNu * JNuNvx * J;
				addGradients(gradVi, &input.d_vertexPosGrad[v_index_inner.x].x);

				getJ(J, TR, vk, vi);
				// gradients
				mat1x3 gradVk = GVCB * JCoLi * JLiNo * JNoNu * JNuNvx * J;
				addGradients(gradVk, &input.d_vertexPosGrad[v_index_inner.z].x);
				gradVi = -GVCB * JCoLi * JLiNo * JNoNu * JNuNvx * J;
				addGradients(gradVi, &input.d_vertexPosGrad[v_index_inner.x].x);
			}
		}
	}
}

//==============================================================================================//

extern "C" void renderBuffersGradGPU(CUDABasedRasterizationGradInput& input)
{
	renderBuffersGradDevice << <(input.numberOfCameras*input.w*input.h + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> > (input);
}

//==============================================================================================//
