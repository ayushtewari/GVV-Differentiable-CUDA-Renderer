
//==============================================================================================//

#include <cuda_runtime.h> 
#include "../Utils/cudaUtil.h"
#include "CUDABasedRasterizationInput.h"
#include "../Utils/CameraUtil.h"
#include "../Utils/IndexHelper.h"
#include "../Utils/cuda_SimpleMatrixUtil.h"
#include "../Utils/RendererUtil.h"

#ifndef FLT_MAX
#define FLT_MAX  1000000
#endif

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

		input.d_faceIDBuffer[idx] = -1;

		input.d_barycentricCoordinatesBuffer[2 * idx + 0] = 0.f;
		input.d_barycentricCoordinatesBuffer[2 * idx + 1] = 0.f;

		input.d_renderBuffer[3 * idx + 0] = 0.f;
		input.d_renderBuffer[3 * idx + 1] = 1.f;
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
		int idf = index.y;

		int indexv0 = input.d_facesVertex[idf].x;
		int indexv1 = input.d_facesVertex[idf].y;
		int indexv2 = input.d_facesVertex[idf].z;

		float3 v0 = input.d_vertices[indexv0];
		float3 v1 = input.d_vertices[indexv1];
		float3 v2 = input.d_vertices[indexv2];
	
		input.d_faceNormal[idx] = cross(v1 - v0, v2 - v0);
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
		int idv = index.y;

		int2 verFaceId = input.d_vertexFacesId[idv];
		float3 vertNorm;
		for (int i = verFaceId.x; i<verFaceId.x + verFaceId.y; i++)
		{
			int faceId = input.d_vertexFaces[i];

			if (i == verFaceId.x)
				vertNorm = input.d_faceNormal[faceId];
			else
			{
				vertNorm.x = vertNorm.x + input.d_faceNormal[faceId].x;
				vertNorm.y = vertNorm.y + input.d_faceNormal[faceId].y;
				vertNorm.z = vertNorm.z + input.d_faceNormal[faceId].z;
			}
		}
		input.d_vertexNormal[idx] = vertNorm;
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

				float3 abc = uv2barycentric(pixelCenter1.x, pixelCenter1.y, input.d_vertices[indexv0], input.d_vertices[indexv1], input.d_vertices[indexv2], input.d_inverseExtrinsics + idc * 4, input.d_inverseProjection + idc * 4);
				
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

				float3 abc = uv2barycentric(pixelCenter1.x, pixelCenter1.y, input.d_vertices[indexv0], input.d_vertices[indexv1], input.d_vertices[indexv2], input.d_inverseExtrinsics + idc * 4, input.d_inverseProjection + idc * 4);

				bool isInsideTriangle = (abc.x >= -0.001f) && (abc.y >= -0.001f) && (abc.z >= -0.001f) && (abc.x <= 1.001f) && (abc.y <= 1.001f) && (abc.z <= 1.001f);
	
				float z = 1.f / (abc.x / vertex0.z + abc.y / vertex1.z + abc.z / vertex2.z); //Perspective-Correct Interpolation

				int pixelId = idc* input.w* input.h + input.w * v + u;

				if (isInsideTriangle && (int)z == input.d_depthBuffer[pixelId])
				{
					int pixelId1 = 2 * idc* input.w * input.h + 2 * input.w * v + 2 * u;
					int pixelId2 = 3 * idc* input.w * input.h + 3 * input.w * v + 3 * u;

					//face buffer
					input.d_faceIDBuffer[pixelId] = idf;
				
					//barycentric buffer
					input.d_barycentricCoordinatesBuffer[pixelId1 + 0] = abc.x;
					input.d_barycentricCoordinatesBuffer[pixelId1 + 1] = abc.y;

					//get pix normal
					float3 v0_norm = input.d_vertexNormal[input.N*idc + indexv0];
					float3 v1_norm = input.d_vertexNormal[input.N*idc + indexv1];
					float3 v2_norm = input.d_vertexNormal[input.N*idc + indexv2];
					float3 pixNorm = v0_norm * abc.x + v1_norm * abc.y + v2_norm * abc.z;
					pixNorm = pixNorm / length(pixNorm);

					//get normal flip
					float3 o = make_float3(0.f, 0.f, 0.f);
					float3 d = make_float3(0.f, 0.f, 0.f);
					getRayCuda2(pixelCenter1, o, d, input.d_inverseExtrinsics + idc * 4, input.d_inverseProjection + idc * 4);
					if (dot(pixNorm, d) > 0.f)
						pixNorm = -pixNorm;

					float3 color = make_float3(0.f,0.f,0.f);

					//albedo
					if (input.albedoMode == AlbedoMode::Textured)
					{
						float2 texCoord0 = make_float2(input.d_textureCoordinates[idf * 3 * 2 + 0 * 2 + 0], 1.f - input.d_textureCoordinates[idf * 3 * 2 + 0 * 2 + 1]);
						float2 texCoord1 = make_float2(input.d_textureCoordinates[idf * 3 * 2 + 1 * 2 + 0], 1.f - input.d_textureCoordinates[idf * 3 * 2 + 1 * 2 + 1]);
						float2 texCoord2 = make_float2(input.d_textureCoordinates[idf * 3 * 2 + 2 * 2 + 0], 1.f - input.d_textureCoordinates[idf * 3 * 2 + 2 * 2 + 1]);
						float2 finalTexCoord = texCoord0* abc.x + texCoord1* abc.y + texCoord2* abc.z;
						finalTexCoord.x = finalTexCoord.x * input.texWidth;
						finalTexCoord.y = finalTexCoord.y * input.texHeight;

						finalTexCoord.x = fmaxf(finalTexCoord.x, 0);
						finalTexCoord.x = fminf(finalTexCoord.x, input.texWidth - 1);
						finalTexCoord.y = fmaxf(finalTexCoord.y, 0);
						finalTexCoord.y = fminf(finalTexCoord.y, input.texHeight - 1);
						 
						float U0 = finalTexCoord.x;
						float V0 = finalTexCoord.y;

						float  LU = int(finalTexCoord.x - 0.5f) + 0.5f;
						float  HU = int(finalTexCoord.x - 0.5f) + 1.5f;

						float  LV = int(finalTexCoord.y - 0.5f) + 0.5f;
						float  HV = int(finalTexCoord.y - 0.5f) + 1.5f;

						float3 colorLULV = make_float3(
							input.d_textureMap[3 * input.texWidth *(int)LV + 3 * (int)LU + 0],
							input.d_textureMap[3 * input.texWidth *(int)LV + 3 * (int)LU + 1],
							input.d_textureMap[3 * input.texWidth *(int)LV + 3 * (int)LU + 2]);

						float3 colorLUHV = make_float3(
							input.d_textureMap[3 * input.texWidth *(int)HV + 3 * (int)LU + 0],
							input.d_textureMap[3 * input.texWidth *(int)HV + 3 * (int)LU + 1],
							input.d_textureMap[3 * input.texWidth *(int)HV + 3 * (int)LU + 2]);

						float3 colorHULV = make_float3(
							input.d_textureMap[3 * input.texWidth *(int)LV + 3 * (int)HU + 0],
							input.d_textureMap[3 * input.texWidth *(int)LV + 3 * (int)HU + 1],
							input.d_textureMap[3 * input.texWidth *(int)LV + 3 * (int)HU + 2]);

						float3 colorHUHV = make_float3(
							input.d_textureMap[3 * input.texWidth *(int)HV + 3 * (int)HU + 0],
							input.d_textureMap[3 * input.texWidth *(int)HV + 3 * (int)HU + 1],
							input.d_textureMap[3 * input.texWidth *(int)HV + 3 * (int)HU + 2]);

						color = (V0 - LV) * (((U0 - LU) * colorLULV) + ((HU - U0) * colorHULV)) +
							    (HV - V0) * (((U0 - LU) * colorLUHV) + ((HU - U0) * colorHUHV));
					}
					else if (input.albedoMode == AlbedoMode::VertexColor)
					{
						color = make_float3(
							input.d_vertexColor[indexv0].x * abc.x + input.d_vertexColor[indexv1].x * abc.y + input.d_vertexColor[indexv2].x * abc.z,
							input.d_vertexColor[indexv0].y * abc.x + input.d_vertexColor[indexv1].y * abc.y + input.d_vertexColor[indexv2].y * abc.z,
							input.d_vertexColor[indexv0].z * abc.x + input.d_vertexColor[indexv1].z * abc.y + input.d_vertexColor[indexv2].z * abc.z);
					}
					else if (input.albedoMode == AlbedoMode::Normal)
					{
						color = make_float3((1.f + pixNorm.x) / 2.f,  (1.f + pixNorm.y) / 2.f, (1.f + pixNorm.z) / 2.f);
					}
					else if (input.albedoMode == AlbedoMode::Lighting)
					{
						color = make_float3(1.f, 1.f, 1.f);
					}
					else if (input.albedoMode == AlbedoMode::ForegroundMask)
					{
						color = make_float3(1.f, 1.f, 1.f);
					}
					
					//shading
					if ((input.shadingMode == ShadingMode::Shaded && (input.albedoMode != AlbedoMode::Normal)) || input.albedoMode == AlbedoMode::Lighting)
					{
						color = getShading(color, pixNorm, input.d_shCoeff + (idc * 27));
					}

					input.d_renderBuffer[pixelId2 + 0] = color.x;
					input.d_renderBuffer[pixelId2 + 1] = color.y;
					input.d_renderBuffer[pixelId2 + 2] = color.z;
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
