//==============================================================================================//
// Classname:
//      GradUtil 
//
//==============================================================================================//
// Description:
//      Basic operations for gradients for rendering 
//
//==============================================================================================//

#pragma once 

//==============================================================================================//

#include <cutil_inline.h>
#include <cutil_math.h>

//==============================================================================================//

__inline__ __device__ void getJCoAl(mat3x3 &JCoAl, float3 pixLight)
{
	JCoAl.setZero();
	JCoAl(0,0) = pixLight.x;
	JCoAl(1,1) = pixLight.y;
	JCoAl(2,2) = pixLight.z;
}

__inline__ __device__ void getJAlVc(mat3x9 &JAlVc, float3 bcc)
{
	JAlVc.setZero();
	JAlVc(0,0) = bcc.x;
	JAlVc(0,3) = bcc.y;
	JAlVc(0,6) = bcc.z;
	JAlVc(1,1) = bcc.x;
	JAlVc(1,4) = bcc.y;
	JAlVc(1,7) = bcc.z;
	JAlVc(2,2) = bcc.x;
	JAlVc(2,5) = bcc.y;
	JAlVc(2,8) = bcc.z;
}

__inline__ __device__ void getJCoLi(mat3x3 &JCoLi, float3 pixAlb)
{
	JCoLi.setZero();
	JCoLi(0,0) = pixAlb.x;
	JCoLi(1,1) = pixAlb.y;
	JCoLi(2,2) = pixAlb.z;
}

__inline__ __device__ float3 getIllum(float3 dir, const float *shCoeffs)
{
	float3 dirSq = dir * dir;
	float3 light;

	light.x = shCoeffs[0];
	light.x += shCoeffs[1] * dir.y;
	light.x += shCoeffs[2] * dir.z;
	light.x += shCoeffs[3] * dir.x;
	light.x += shCoeffs[4] * (dir.x * dir.y);
	light.x += shCoeffs[5] * (dir.z * dir.y);
	light.x += shCoeffs[6] * (3 * dirSq.z - 1);
	light.x += shCoeffs[7] * (dir.x * dir.z);
	light.x += shCoeffs[8] * (dirSq.x - dirSq.y);

	light.y = shCoeffs[9 + 0];
	light.y += shCoeffs[9 + 1] * dir.y;
	light.y += shCoeffs[9 + 2] * dir.z;
	light.y += shCoeffs[9 + 3] * dir.x;
	light.y += shCoeffs[9 + 4] * (dir.x * dir.y);
	light.y += shCoeffs[9 + 5] * (dir.z * dir.y);
	light.y += shCoeffs[9 + 6] * (3 * dirSq.z - 1);
	light.y += shCoeffs[9 + 7] * (dir.x * dir.z);
	light.y += shCoeffs[9 + 8] * (dirSq.x - dirSq.y);

	light.z = shCoeffs[18 + 0];
	light.z += shCoeffs[18 + 1] * dir.y;
	light.z += shCoeffs[18 + 2] * dir.z;
	light.z += shCoeffs[18 + 3] * dir.x;
	light.z += shCoeffs[18 + 4] * (dir.x * dir.y);
	light.z += shCoeffs[18 + 5] * (dir.z * dir.y);
	light.z += shCoeffs[18 + 6] * (3 * dirSq.z - 1);
	light.z += shCoeffs[18 + 7] * (dir.x * dir.z);
	light.z += shCoeffs[18 + 8] * (dirSq.x - dirSq.y);
	return light;
}

__inline__ __device__ void getJLiGm(mat3x9 &JLiGm, int rgb, float3 pixNorm)
{
	JLiGm.setZero();

	JLiGm(rgb,0) = 1;
	JLiGm(rgb,1) = pixNorm.y;
	JLiGm(rgb,2) = pixNorm.z;
	JLiGm(rgb,3) = pixNorm.x;
	JLiGm(rgb,4) = pixNorm.x * pixNorm.y;
	JLiGm(rgb,5) = pixNorm.z * pixNorm.y;
	JLiGm(rgb,6) = 3*pixNorm.z*pixNorm.z - 1;
	JLiGm(rgb,7) = pixNorm.x * pixNorm.z;
	JLiGm(rgb,8) = ( (pixNorm.x * pixNorm.x) - (pixNorm.y*pixNorm.y) );
}

__inline__ __device__ void getJLiNo(mat3x3 &JLiNo, float3 dir, float* shCoeff)
{
	JLiNo.setZero();
	for (int i = 0; i<3; i++) {
		JLiNo(i, 0) = shCoeff[(i * 9) + 3] +
			(shCoeff[(i * 9) + 4] * dir.y) +
			(shCoeff[(i * 9) + 7] * dir.z) +
			(shCoeff[(i * 9) + 8] * 2 * dir.x);

		JLiNo(i, 1) = shCoeff[(i * 9) + 0] +
			(shCoeff[(i * 9) + 4] * dir.x) +
			(shCoeff[(i * 9) + 5] * dir.z) +
			(shCoeff[(i * 9) + 8] * -2 * dir.y);

		JLiNo(i, 2) = shCoeff[(i * 9) + 2] +
			(shCoeff[(i * 9) + 5] * dir.y) +
			(shCoeff[(i * 9) + 6] * 6 * dir.z) +
			(shCoeff[(i * 9) + 7] * dir.x);
	}
}

__inline__ __device__ void getJNoNu(mat3x3 &JNoNu, float3 un_vec, float norm)
{
	float norm_p2 = norm * norm;
	float norm_p3 = norm_p2 * norm;

	JNoNu(0, 0) = (norm_p2 - (un_vec.x*un_vec.x)) / (norm_p3);
	JNoNu(1, 1) = (norm_p2 - (un_vec.y*un_vec.y)) / (norm_p3);
	JNoNu(2, 2) = (norm_p2 - (un_vec.z*un_vec.z)) / (norm_p3);

	JNoNu(0, 1) = -(un_vec.x*un_vec.y) / norm_p3;
	JNoNu(1, 0) = JNoNu(0, 1);

	JNoNu(0, 2) = -(un_vec.x*un_vec.z) / norm_p3;
	JNoNu(2, 0) = JNoNu(0, 2);

	JNoNu(1, 2) = -(un_vec.y*un_vec.z) / norm_p3;
	JNoNu(2, 1) = JNoNu(1, 2);
}

__inline__ __device__ void getJ(mat3x3 &J, mat3x3 TR, mat3x1 vj, mat3x1 vi)
{
	float3 temp3;
	mat3x1 Ix(make_float3(1, 0, 0)), Iy(make_float3(0, 1, 0)), Iz(make_float3(0, 0, 1));
	Ix = TR * Ix;
	Iy = TR * Iy;
	Iz = TR * Iz;

	mat3x1 diff = vj - vi;
	temp3 = cross(-Ix, diff);
	J(0, 0) = -temp3.x;		//for adjacent vertex
	J(1, 0) = -temp3.y;
	J(2, 0) = -temp3.z;
	
	temp3 = cross(-Iy, diff);
	J(0, 1) = -temp3.x;
	J(1, 1) = -temp3.y;
	J(2, 1) = -temp3.z;
	
	temp3 = cross(-Iz, diff);
	J(0, 2) = -temp3.x;
	J(1, 2) = -temp3.y;
	J(2, 2) = -temp3.z;
}

__inline__ __device__ void addGradients(mat1x3 grad, float* d_grad)
{
	atomicAdd(&d_grad[0], grad(0,0));
	atomicAdd(&d_grad[1], grad(0,1));
	atomicAdd(&d_grad[2], grad(0,2));
}

__inline__ __device__ void addGradients9(mat1x9 grad, float* d_grad)
{
	for(int ii=0; ii<9; ii++)
		atomicAdd(&d_grad[ii], grad(0,ii));
}

__inline__ __device__ void addGradients9I(mat1x9 grad, float3* d_grad, int3 index)
{
	atomicAdd(&d_grad[index.x*3].x, grad(0,0));
	atomicAdd(&d_grad[index.x*3].y, grad(0,1));
	atomicAdd(&d_grad[index.x*3].z, grad(0,2));
	atomicAdd(&d_grad[index.y*3].x, grad(0,3));
	atomicAdd(&d_grad[index.y*3].y, grad(0,4));
	atomicAdd(&d_grad[index.y*3].z, grad(0,5));
	atomicAdd(&d_grad[index.z*3].x, grad(0,6));
	atomicAdd(&d_grad[index.z*3].y, grad(0,7));
	atomicAdd(&d_grad[index.z*3].z, grad(0,8));
}

__device__ inline mat3x3 getRotationMatrix(float4* d_T)
{
	mat3x3 TE;
	TE(0,0) = d_T[0].x; 
	TE(0,1) = d_T[0].y; 
	TE(0,2) = d_T[0].z; 
	TE(1,0) = d_T[1].x; 
	TE(1,1) = d_T[1].y; 
	TE(1,2) = d_T[1].z; 
	TE(2,0) = d_T[2].x; 
	TE(2,1) = d_T[2].y; 
	TE(2,2) = d_T[2].z; 
	return TE;
}
