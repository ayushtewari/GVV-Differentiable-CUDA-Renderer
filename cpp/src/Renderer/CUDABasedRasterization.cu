
//==============================================================================================//

#include <cuda_runtime.h> 
#include "../Utils/cudaUtil.h"
#include "CUDABasedRasterizationInput.h"
#include "../Utils/CameraUtil.h"
#include "../Utils/IndexHelper.h"

#ifndef FLT_MAX
#define FLT_MAX  1000000
#endif

//==============================================================================================//
//Helpers
//==============================================================================================//

inline __device__ float3 uv2barycentric(float u, float v, float3 v0, float3 v1, float3 v2)
{
	float e2 = ((v0.y - v1.y)*u + (v1.x - v0.x)*v + v0.x*v1.y - v1.x*v0.y) / ((v0.y - v1.y)*v2.x + (v1.x - v0.x)*v2.y + v0.x*v1.y - v1.x*v0.y);
	float e1 = ((v0.y - v2.y)*u + (v2.x - v0.x)*v + v0.x*v2.y - v2.x*v0.y) / ((v0.y - v2.y)*v1.x + (v2.x - v0.x)*v1.y + v0.x*v2.y - v2.x*v0.y);
	float e0 = 1.f - e2 - e1;
	return make_float3(e0, e1, e2);
}

//==============================================================================================//

inline __device__ float3 getShading(float3 color, float3 dir, const float *shCoeffs)
{
	float3 dirSq = dir * dir;
	float3 shadedColor;

	shadedColor.x = shCoeffs[0];
	shadedColor.x += shCoeffs[1] * dir.y;
	shadedColor.x += shCoeffs[2] * dir.z;
	shadedColor.x += shCoeffs[3] * dir.x;
	shadedColor.x += shCoeffs[4] * (dir.x * dir.y);
	shadedColor.x += shCoeffs[5] * (dir.z * dir.y);
	shadedColor.x += shCoeffs[6] * (3 * dirSq.z - 1);
	shadedColor.x += shCoeffs[7] * (dir.x * dir.z);
	shadedColor.x += shCoeffs[8] * (dirSq.x - dirSq.y);
	shadedColor.x = shadedColor.x * color.x;

	shadedColor.y = shCoeffs[9 + 0];
	shadedColor.y += shCoeffs[9 + 1] * dir.y;
	shadedColor.y += shCoeffs[9 + 2] * dir.z;
	shadedColor.y += shCoeffs[9 + 3] * dir.x;
	shadedColor.y += shCoeffs[9 + 4] * (dir.x * dir.y);
	shadedColor.y += shCoeffs[9 + 5] * (dir.z * dir.y);
	shadedColor.y += shCoeffs[9 + 6] * (3 * dirSq.z - 1);
	shadedColor.y += shCoeffs[9 + 7] * (dir.x * dir.z);
	shadedColor.y += shCoeffs[9 + 8] * (dirSq.x - dirSq.y);
	shadedColor.y = shadedColor.y * color.y;

	shadedColor.z = shCoeffs[18 + 0];
	shadedColor.z += shCoeffs[18 + 1] * dir.y;
	shadedColor.z += shCoeffs[18 + 2] * dir.z;
	shadedColor.z += shCoeffs[18 + 3] * dir.x;
	shadedColor.z += shCoeffs[18 + 4] * (dir.x * dir.y);
	shadedColor.z += shCoeffs[18 + 5] * (dir.z * dir.y);
	shadedColor.z += shCoeffs[18 + 6] * (3 * dirSq.z - 1);
	shadedColor.z += shCoeffs[18 + 7] * (dir.x * dir.z);
	shadedColor.z += shCoeffs[18 + 8] * (dirSq.x - dirSq.y);
	shadedColor.z = shadedColor.z * color.z;
	return shadedColor;
}

//==============================================================================================//
//Render buffers
//==============================================================================================//

/*
Initializes all arrays
*/
__global__ void initializeDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx<input.w*input.h*input.numberOfCameras)
	{
		input.d_depthBuffer[idx] = INT_MAX;

		input.d_faceIDBuffer[idx * 4 + 0] = -1;
		input.d_faceIDBuffer[idx * 4 + 1] = -1;
		input.d_faceIDBuffer[idx * 4 + 2] = -1;
		input.d_faceIDBuffer[idx * 4 + 3] = -1;

		input.d_barycentricCoordinatesBuffer[3 * idx + 0] = 0.f;
		input.d_barycentricCoordinatesBuffer[3 * idx + 1] = 0.f;
		input.d_barycentricCoordinatesBuffer[3 * idx + 2] = 0.f;

		input.d_renderBuffer[3 * idx + 0] = 0.f;
		input.d_renderBuffer[3 * idx + 1] = 0.f;
		input.d_renderBuffer[3 * idx + 2] = 0.f;
	}
}

//==============================================================================================//

/*
Project the vertices into the image plane and store depth value
*/
__global__ void projectVerticesDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input.numberOfCameras * input.N)
	{
		int2 index = index1DTo2D(input.numberOfCameras, input.N, idx);
		int idc = index.x;
		int idv = index.y;

		float3 v0 = input.d_vertices[idv];

		float3 c_v0 = getCamSpacePoint(&input.d_cameraExtrinsics[3 * idc], v0);
		float3 i_v0 = projectPointFloat3(&input.d_cameraIntrinsics[3 * idc], c_v0);

		input.d_projectedVertices[idx] = i_v0;
	}
}

//==============================================================================================//

/*
Computes the face normals
*/
__global__ void renderFaceNormalDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input.numberOfCameras * input.F)
	{
		int2 index = index1DTo2D(input.numberOfCameras, input.F, idx);
		int idc = index.x;
		int idf = index.y;

		int indexv0 = input.d_facesVertex[idf].x;
		int indexv1 = input.d_facesVertex[idf].y;
		int indexv2 = input.d_facesVertex[idf].z;

		float3 v0 = input.d_vertices[indexv0];
		float3 v1 = input.d_vertices[indexv1];
		float3 v2 = input.d_vertices[indexv2];

		float3 c_v0 = getCamSpacePoint(&input.d_cameraExtrinsics[3 * idc], v0);
		float3 c_v1 = getCamSpacePoint(&input.d_cameraExtrinsics[3 * idc], v1);
		float3 c_v2 = getCamSpacePoint(&input.d_cameraExtrinsics[3 * idc], v2);

		input.d_faceNormal[idx] = cross(c_v1 - c_v0, c_v2 - c_v0);
	}
}

//==============================================================================================//

/*
Computes the vertex normals
*/
__global__ void renderVertexNormalDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input.numberOfCameras * input.N)
	{
		int2 index = index1DTo2D(input.numberOfCameras, input.N, idx);
		int idc = index.x;
		int idv = index.y;

		int2 verFaceId = input.d_vertexFacesId[idv];
		float3 vertNorm= make_float3(0.f,0.f,0.f);
		float faceCounter = 0.f;
		for (int i = verFaceId.x; i<verFaceId.x + verFaceId.y; i++)
		{
			int faceId = input.d_vertexFaces[i];
			vertNorm = vertNorm + input.d_faceNormal[faceId];
			faceCounter++;
		}
		input.d_vertexNormal[idx] = vertNorm/faceCounter;
	}
}

//==============================================================================================//

/*
Project the vertices into the image plane,
computes the 2D bounding box per triangle in the image plane
and computes the maximum bounding box for all triangles of the mesh
*/
__global__ void projectFacesDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input.numberOfCameras * input.F)
	{
		int2 index = index1DTo2D(input.numberOfCameras, input.F, idx);
		int idc = index.x;
		int idf = index.y;

		int indexv0 = input.d_facesVertex[idf].x;
		int indexv1 = input.d_facesVertex[idf].y;
		int indexv2 = input.d_facesVertex[idf].z;

		float3 i_v0 = input.d_projectedVertices[idc* input.N + indexv0];
		float3 i_v1 = input.d_projectedVertices[idc* input.N + indexv1];
		float3 i_v2 = input.d_projectedVertices[idc* input.N + indexv2];

		input.d_BBoxes[idx].x = fmaxf(fminf(i_v0.x, fminf(i_v1.x, i_v2.x)) - 0.5f, 0);  //minx
		input.d_BBoxes[idx].y = fmaxf(fminf(i_v0.y, fminf(i_v1.y, i_v2.y)) - 0.5f, 0);  //miny

		input.d_BBoxes[idx].z = fminf(fmaxf(i_v0.x, fmaxf(i_v1.x, i_v2.x)) + 0.5f, input.w - 1);   //maxx
		input.d_BBoxes[idx].w = fminf(fmaxf(i_v0.y, fmaxf(i_v1.y, i_v2.y)) + 0.5f, input.h - 1);  //maxy
	}
}

//==============================================================================================//

/*
Render the depth, faceId and barycentricCoordinates buffers
*/
__global__ void renderDepthBufferDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input.numberOfCameras * input.F)
	{
		int2 index = index1DTo2D(input.numberOfCameras, input.F, idx);
		int idc = index.x;
		int idf = index.y;

		int indexv0 = input.d_facesVertex[idf].x;
		int indexv1 = input.d_facesVertex[idf].y;
		int indexv2 = input.d_facesVertex[idf].z;

		float3 vertex0 = input.d_projectedVertices[input.N*idc + indexv0];
		float3 vertex1 = input.d_projectedVertices[input.N*idc + indexv1];
		float3 vertex2 = input.d_projectedVertices[input.N*idc + indexv2];

		for (int u = input.d_BBoxes[idx].x; u <= input.d_BBoxes[idx].z; u++)
		{
			for (int v = input.d_BBoxes[idx].y; v <= input.d_BBoxes[idx].w; v++)
			{
				float2 pixelCenter1 = make_float2(u + 0.5f, v + 0.5f);

				float3 abc = uv2barycentric(pixelCenter1.x, pixelCenter1.y, vertex0, vertex1, vertex2);
				float z = FLT_MAX;
				bool isInsideTriangle = (abc.x >= -0.001f) && (abc.y >= -0.001f) && (abc.z >= -0.001f) && (abc.x <= 1.001f) && (abc.y <= 1.001f) && (abc.z <= 1.001f);

				if (isInsideTriangle)
				{
					z = 1.f / (abc.x / vertex0.z + abc.y / vertex1.z + abc.z / vertex2.z); //Perspective-Correct Interpolation

					int pixelId = idc* input.w* input.h + input.w * v + u;
					atomicMin(&input.d_depthBuffer[pixelId], z);
				}
			}
		}
	}
}

//==============================================================================================//

/*
Render the faceId and barycentricCoordinates buffers
*/
__global__ void renderBuffersDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input.numberOfCameras * input.F)
	{
		int2 index = index1DTo2D(input.numberOfCameras, input.F, idx);
		int idc = index.x;
		int idf = index.y;

		int indexv0 = input.d_facesVertex[idf].x;
		int indexv1 = input.d_facesVertex[idf].y;
		int indexv2 = input.d_facesVertex[idf].z;

		float3 vertex0 = input.d_projectedVertices[input.N*idc + indexv0];
		float3 vertex1 = input.d_projectedVertices[input.N*idc + indexv1];
		float3 vertex2 = input.d_projectedVertices[input.N*idc + indexv2];

		for (int u = input.d_BBoxes[idx].x; u <= input.d_BBoxes[idx].z; u++)
		{
			for (int v = input.d_BBoxes[idx].y; v <= input.d_BBoxes[idx].w; v++)
			{
				float2 pixelCenter1 = make_float2(u + 0.5f, v + 0.5f);

				float3 abc = uv2barycentric(pixelCenter1.x, pixelCenter1.y, vertex0, vertex1, vertex2);

				bool isInsideTriangle = (abc.x >= -0.001f) && (abc.y >= -0.001f) && (abc.z >= -0.001f) && (abc.x <= 1.001f) && (abc.y <= 1.001f) && (abc.z <= 1.001f);

				float z = 1.f / (abc.x / vertex0.z + abc.y / vertex1.z + abc.z / vertex2.z); //Perspective-Correct Interpolation
				float3 ABC = make_float3(z*abc.x / vertex0.z, z*abc.y / vertex1.z, z*abc.z / vertex2.z);
				abc = ABC;

				int pixelId = idc* input.w* input.h + input.w * v + u;

				if (isInsideTriangle && (int)z == input.d_depthBuffer[pixelId])
				{
					int pixelId2 = 3 * idc* input.w * input.h + 3 * input.w * v + 3 * u;
					int pixelId3 = 4 * idc* input.w * input.h + 4 * input.w * v + 4 * u;

					//face buffer
					input.d_faceIDBuffer[pixelId3 + 0] = idf;
					input.d_faceIDBuffer[pixelId3 + 1] = indexv0;
					input.d_faceIDBuffer[pixelId3 + 2] = indexv1;
					input.d_faceIDBuffer[pixelId3 + 3] = indexv2;

					//barycentric buffer
					input.d_barycentricCoordinatesBuffer[pixelId2 + 0] = abc.x;
					input.d_barycentricCoordinatesBuffer[pixelId2 + 1] = abc.y;
					input.d_barycentricCoordinatesBuffer[pixelId2 + 2] = abc.z;


					//shading
					float3 v0_norm = input.d_vertexNormal[input.N*idc + indexv0];
					float3 v1_norm = input.d_vertexNormal[input.N*idc + indexv1];
					float3 v2_norm = input.d_vertexNormal[input.N*idc + indexv2];
					float3 pixNorm = v0_norm * abc.x + v1_norm * abc.y + v2_norm * abc.z;
					float pixNormNorm = sqrtf(pixNorm.x*pixNorm.x + pixNorm.y*pixNorm.y + pixNorm.z*pixNorm.z);
					pixNorm = pixNorm / pixNormNorm;

					//render buffer
					if (input.renderMode == RenderMode::Textured)
					{
						float2 texCoord0 = make_float2(input.d_textureCoordinates[idf * 3 * 2 + 0 * 2 + 0], input.d_textureCoordinates[idf * 3 * 2 + 0 * 2 + 1]);
						float2 texCoord1 = make_float2(input.d_textureCoordinates[idf * 3 * 2 + 1 * 2 + 0], input.d_textureCoordinates[idf * 3 * 2 + 1 * 2 + 1]);
						float2 texCoord2 = make_float2(input.d_textureCoordinates[idf * 3 * 2 + 2 * 2 + 0], input.d_textureCoordinates[idf * 3 * 2 + 2 * 2 + 1]);
						float2 finalTexCoord = texCoord0* abc.x + texCoord1* abc.y + texCoord2* abc.z;
						finalTexCoord.x = finalTexCoord.x * input.texWidth;
						finalTexCoord.y = (1.f - finalTexCoord.y) * input.texHeight;

						finalTexCoord.x = fmaxf(finalTexCoord.x, 0);
						finalTexCoord.x = fminf(finalTexCoord.x, input.texWidth - 1);
						finalTexCoord.y = fmaxf(finalTexCoord.y, 0);
						finalTexCoord.y = fminf(finalTexCoord.y, input.texHeight - 1);

						float3 color = make_float3(input.d_textureMap[3 * input.texWidth *(int)finalTexCoord.y + 3 * (int)finalTexCoord.x + 0],
							input.d_textureMap[3 * input.texWidth *(int)finalTexCoord.y + 3 * (int)finalTexCoord.x + 1],
							input.d_textureMap[3 * input.texWidth *(int)finalTexCoord.y + 3 * (int)finalTexCoord.x + 2]);

						float3 colorShaded = getShading(color, pixNorm, input.d_shCoeff + (idc * 27));
						input.d_renderBuffer[pixelId2 + 0] = colorShaded.x;
						input.d_renderBuffer[pixelId2 + 1] = colorShaded.y;
						input.d_renderBuffer[pixelId2 + 2] = colorShaded.z;
					}
					else if (input.renderMode == RenderMode::VertexColor)
					{
						//vertex color buffer
						float3 color = make_float3(
							input.d_vertexColor[indexv0].x * abc.x + input.d_vertexColor[indexv1].x * abc.y + input.d_vertexColor[indexv2].x * abc.z,
							input.d_vertexColor[indexv0].y * abc.x + input.d_vertexColor[indexv1].y * abc.y + input.d_vertexColor[indexv2].y * abc.z,
							input.d_vertexColor[indexv0].z * abc.x + input.d_vertexColor[indexv1].z * abc.y + input.d_vertexColor[indexv2].z * abc.z);

						float3 colorShaded = getShading(color, pixNorm, input.d_shCoeff + (idc * 27));
						input.d_renderBuffer[pixelId2 + 0] = colorShaded.x;
						input.d_renderBuffer[pixelId2 + 1] = colorShaded.y;
						input.d_renderBuffer[pixelId2 + 2] = colorShaded.z;
					}
				}
			}
		}
	}
}

//==============================================================================================//

extern "C" void renderBuffersGPU(CUDABasedRasterizationInput& input)
{
	initializeDevice			<< <(input.w*input.h*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> > (input);

	projectVerticesDevice		<< <(input.N*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input);

	projectFacesDevice			<< <(input.F*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input);

	renderFaceNormalDevice		<< <(input.F*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input);

	renderVertexNormalDevice	<< <(input.N*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input);

	renderDepthBufferDevice		<< <(input.F*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input);

	renderBuffersDevice			<< <(input.F*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input);
}
