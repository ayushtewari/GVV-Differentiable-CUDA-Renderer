
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
		int3 index = index1DTo3D(input.numberOfCameras, input.w, input.h, idx);
		int idc    = index.x;
		int idw    = index.y;
		int idh    = index.z;
                int4 idf   = input.d_faceIDBuffer[idx];

		// return cases
		if(idf.w == -1)
			return;
		//------------------------------------------------

		// Co = Al * Li
		// 	Al = Bc * Vc
		// 		Bc = f(Vp)
		// 	Li = Gm * No
		// 		No = Noun / norm
		//			Noun = f(Bc, Nv)
		// 				Nv = f(Nf)
		// 					Nf = f(Vp)
		// 				Bc = f(Vp)
				
		//------------------------------------------------

		// JCoAl [3x3]                                       
		// 	JAlVc [3x9]                                   
		// 	JAlBc [3x3]
		// 		JBcVp [3x9]
		// JCoLi [3x3]
		// 	JLiGm [3x27]
		// 	JLiNo [3x3]
		//		JNoNu [3x3]
		//			JNuNv0 [3x3], JNuNv1 [3x3], JNuNv2 [3x3]
		//				JNvVp [3x3 for each (Nvx, N(Nv))] 
		//			JNuBc
		//				JBcVp



		// 		JNoBc [3x3]
		// 			JBcVp [3x9]
		// 		JNoNv [3x9]
		// 			JNvNf [9x3y]
		// 				JNfVp [3yx3z]

		//------------------------------------------------

		//------------------------------------------------

		// Co = Al * Li
		// 	Al = Bc * Vc
		// 		Bc = f(Vp)
		// 	Li = Gm * No
		// 		No = f(Bc, Nv)
		// 			Nv = f(Nf)
		// 				Nf = f(Vp)
		// 			Bc = f(Vp)
				
		//------------------------------------------------

		// JCoAl [3x3]                                       
		// 	JAlVc [3x9]                                   
		// 	JAlBc [3x3]
		// 		JBcVp [3x9]
		// JCoLi [3x3]
		// 	JLiGm [3x27]
		// 	JLiNo [3x3]
		// 		JNoBc [3x3]
		// 			JBcVp [3x9]
		// 		JNoNv [3x9]
		// 			JNvNf [9x3y]
		// 				JNfVp [3yx3z]

		//------------------------------------------------

		// gradVerPos = diag(Li) * (JCoAl * JAlBc * JBcVp) + diag(Al) * (JCoLi * JLiNo * JNoBc * JBcVp)
		// gradVerCol = JCoAl * JAlVc 

		//------------------------------------------------

		// data initialization 1
		float3 bcc               =      input.d_barycentricCoordinatesBuffer[idx];
		int3   v_index           =      input.d_facesVertex[idf.x]; 
		const float* shCoeff     =      input.d_shCoeff + idc*27;

		// data initialization 2
		float3 vertexPos0        =	input.d_vertices[v_index.x];
		float3 vertexPos1        =	input.d_vertices[v_index.y];
		float3 vertexPos2        =	input.d_vertices[v_index.z];
		float3 vertexCol0        =      input.d_vertexColor[v_index.x];
		float3 vertexCol1        =      input.d_vertexColor[v_index.y];
		float3 vertexCol2        =      input.d_vertexColor[v_index.z];
		float3 vertexNor0        =      input.d_vertexNormal[idc*input.N + v_index.x];
		float3 vertexNor1        =      input.d_vertexNormal[idc*input.N + v_index.y];
		float3 vertexNor2        =      input.d_vertexNormal[idc*input.N + v_index.z];
		
		// data holders
		mat1x3 GVCB; 
		mat3x3 JCoAl, JCoLi, JLiNo, JNoNu, JNuNv, JNuNvx, J, TR;
		mat3x9 JAlVc, JBcVp;
		mat3x9 JLiGmR, JLiGmG, JLiGmB;

		// 
		float3 pixAlb            = bcc.x * vertexCol0 + bcc.y * vertexCol1 + bcc.z * vertexCol2; 
		float3 pixNormUn         = bcc.x * vertexNor0 + bcc.y * vertexNor1 + bcc.z * vertexNor2;
		float  pixNormVal        = (sqrtf(pixNormUn.x*pixNormUn.x + pixNormUn.y*pixNormUn.y + pixNormUn.z*pixNormUn.z));
		float3 pixNorm           = pixNormUn/pixNormVal;
		
		// 
		float3 pixLight          = getIllum(pixNorm, shCoeff);

		//
		GVCB(0,0) = input.d_vertexColorBufferGrad[idx].x; GVCB(0,1) = input.d_vertexColorBufferGrad[idx].y; GVCB(0,2) = input.d_vertexColorBufferGrad[idx].z;
		 
		// jacobians
		getJCoAl(JCoAl, pixLight);
		getJAlVc(JAlVc, bcc); 
		getJCoLi(JCoLi, pixAlb);
		getJLiGm(JLiGmR, 0, pixNorm);
		getJLiGm(JLiGmG, 1, pixNorm);
		getJLiGm(JLiGmB, 2, pixNorm);
		getJLiNo(JLiNo, pixNorm, (float *)shCoeff);
		getJNoNu(JNoNu, pixNormUn, pixNormVal);

		// Gradients
		mat1x9 gradVerCol   = GVCB * JCoAl * JAlVc;
		mat1x9 gradSHCoeffR = GVCB * JCoLi * JLiGmR;
		mat1x9 gradSHCoeffG = GVCB * JCoLi * JLiGmG;
		mat1x9 gradSHCoeffB = GVCB * JCoLi * JLiGmB;
		//mat1x9 gradVerPos   = GVCB * JCoAl * JAlBc * JBcVp;
		addGradients9I(gradVerCol, input.d_vertexColorGrad, v_index);
		addGradients9(gradSHCoeffR, &input.d_shCoeffGrad[idc*27]);

		//
		int idv;
		JNuNvx.setIdentity();
		for(int i=0; i<3; i++)
		{
			//
			if(i==0) { idv = v_index.x; JNuNvx = bcc.x * JNuNvx; }
			else if(i==1) { idv = v_index.y;JNuNvx = bcc.y * JNuNvx; }
			else { idv = v_index.z;JNuNvx = bcc.z * JNuNvx; }

			//
			int2 verFaceId = input.d_vertexFacesId[idv];

			//
			mat3x1 vi, vj, vk;
			for (int j = verFaceId.x; j<verFaceId.x + verFaceId.y; j++)
			{
				int faceId          = input.d_vertexFaces[j];
				int3 v_index_inner  = input.d_facesVertex[faceId]; 	
				vi = TR * (mat3x1) input.d_vertices[v_index_inner.x];
				vj = TR * (mat3x1) input.d_vertices[v_index_inner.y];
				vk = TR * (mat3x1) input.d_vertices[v_index_inner.z];

				getJ(J, TR, vj, vi);
				// gradients
				mat1x3 gradVj =  GVCB * JCoLi * JLiNo * JNoNu * JNuNvx * J; 
				addGradients(gradVj, &input.d_vertexPosGrad[v_index_inner.y].x);
				mat1x3 gradVi = -GVCB * JCoLi * JLiNo * JNoNu * JNuNvx * J; 	
				addGradients(gradVi, &input.d_vertexPosGrad[v_index_inner.x].x);

				getJ(J, TR, vk, vi);
				// gradients
				mat1x3 gradVk =  GVCB * JCoLi * JLiNo * JNoNu * JNuNvx * J; 	
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
	renderBuffersGradDevice << <(input.numberOfCameras*input.w*input.h + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input);
}

//==============================================================================================//
