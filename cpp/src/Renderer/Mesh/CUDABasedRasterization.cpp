//==============================================================================================//

#include "CUDABasedRasterization.h"

//==============================================================================================//

#define CLOCKS_PER_SEC ((clock_t)1000) 

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

CUDABasedRasterization::CUDABasedRasterization(trimesh* mesh, camera_container* cm)
	:
	mesh(mesh),
	cameras(cm)
{
	if (mesh == NULL)
	{
		std::cout << "Character not initialized in CUDABasedRasterization!" << std::endl;
	}
	if (cameras == NULL)
	{
		std::cout << "Cameras not initialized in CUDABasedRasterization!" << std::endl;
	}

	if (mesh != NULL && cameras != NULL)
	{
		input.w = cameras->getCamera(0)->getWidth();
		input.h = cameras->getCamera(0)->getHeight();

		h_barycentricCoordinatesBuffer =	new float[3 * input.w * input.h*cameras->getNrCameras()];
		h_faceIDBuffer =					new int4[input.w * input.h*cameras->getNrCameras()];
		h_depthBuffer =						new int[input.w * input.h*cameras->getNrCameras()];
		h_renderBuffer =					new float[3 * input.w * input.h*cameras->getNrCameras()];
		h_vertexColorBuffer =				new float[3 * input.w * input.h*cameras->getNrCameras()];

		cutilSafeCall(cudaMalloc(&input.d_depthBuffer,							sizeof(int)   *		input.w * input.h *	cameras->getNrCameras()));
		cutilSafeCall(cudaMalloc(&input.d_faceIDBuffer,							sizeof(int)   *	4 *	input.w * input.h *	cameras->getNrCameras()));
		cutilSafeCall(cudaMalloc(&input.d_barycentricCoordinatesBuffer,			sizeof(float) * 3 * input.w * input.h * cameras->getNrCameras()));
		cutilSafeCall(cudaMalloc(&input.d_renderBuffer,							sizeof(float) * 3 * input.w * input.h * cameras->getNrCameras()));
		cutilSafeCall(cudaMalloc(&input.d_vertexColorBuffer,					sizeof(float) * 3 * input.w * input.h * cameras->getNrCameras()));

		cutilSafeCall(cudaMalloc(&input.d_BBoxes,								sizeof(int4)*	mesh->F*cameras->getNrCameras()));
		cutilSafeCall(cudaMalloc(&input.d_projectedVertices,					sizeof(float3)*	mesh->N*cameras->getNrCameras()));

		input.d_vertices					= mesh->d_vertices;
		input.d_facesVertex					= mesh->d_facesVertex;
		input.F								= mesh->F;
		input.N								= mesh->N;
		input.numberOfCameras				= cameras->getNrCameras();
		input.d_cameraExtrinsics			= cameras->getD_allCameraExtrinsics();
		input.d_cameraIntrinsics			= cameras->getD_allCameraIntrinsics();

		input.d_visibilities				= mesh->d_visibilities;
		input.d_boundaries					= mesh->d_boundaries;
		input.d_vertexColor					= mesh->d_vertexColors;
		input.d_textureCoordinates			= mesh->d_textureCoordinates;
		input.d_textureMap					= mesh->d_textureMap;
		input.texHeight						= mesh->textureHeight;
		input.texWidth						= mesh->textureWidth;
	}
	else
	{
		std::cout << "Unable to initialize CUDABasedRasterization!" << std::endl;
	}
}

//==============================================================================================//

CUDABasedRasterization::~CUDABasedRasterization()
{
	cutilSafeCall(cudaFree(input.d_depthBuffer));
	cutilSafeCall(cudaFree(input.d_faceIDBuffer));
	cutilSafeCall(cudaFree(input.d_barycentricCoordinatesBuffer));
	cutilSafeCall(cudaFree(input.d_BBoxes));
	cutilSafeCall(cudaFree(input.d_projectedVertices));
}

//==============================================================================================//

void CUDABasedRasterization::renderBuffers()
{
	renderBuffersGPU(input);
}

//==============================================================================================//

void CUDABasedRasterization::checkVisibility(bool checkBoundary)
{
	checkVisibilityGPU(input, checkBoundary);
}

//==============================================================================================//

void  CUDABasedRasterization::copyDepthBufferGPU2CPU()
{
	cutilSafeCall(cudaMemcpy(h_depthBuffer, input.d_depthBuffer, sizeof(int) * input.w * input.h * cameras->getNrCameras(), cudaMemcpyDeviceToHost));
}

//==============================================================================================//

void  CUDABasedRasterization::copyBarycentricBufferGPU2CPU()
{
	cutilSafeCall(cudaMemcpy(h_barycentricCoordinatesBuffer, input.d_barycentricCoordinatesBuffer, sizeof(float) * 3 * cameras->getCamera(0)->getWidth()* cameras->getCamera(0)->getHeight()*cameras->getNrCameras(), cudaMemcpyDeviceToHost));
}

//==============================================================================================//

void  CUDABasedRasterization::copyFaceIdBufferGPU2CPU()
{
	cutilSafeCall(cudaMemcpy(h_faceIDBuffer, input.d_faceIDBuffer, sizeof(int4) *  cameras->getCamera(0)->getWidth()* cameras->getCamera(0)->getHeight()*cameras->getNrCameras(), cudaMemcpyDeviceToHost));
}

//==============================================================================================//

void CUDABasedRasterization::copyRenderBufferGPU2CPU()
{
	cutilSafeCall(cudaMemcpy(h_renderBuffer, input.d_renderBuffer, sizeof(float) * 3 * cameras->getCamera(0)->getWidth()* cameras->getCamera(0)->getHeight()*cameras->getNrCameras(), cudaMemcpyDeviceToHost));
}
//==============================================================================================//

void CUDABasedRasterization::copyVertexColorBufferGPU2CPU()
{
	cutilSafeCall(cudaMemcpy(h_vertexColorBuffer, input.d_vertexColorBuffer, sizeof(float) * 3 * cameras->getCamera(0)->getWidth()* cameras->getCamera(0)->getHeight()*cameras->getNrCameras(), cudaMemcpyDeviceToHost));
}
