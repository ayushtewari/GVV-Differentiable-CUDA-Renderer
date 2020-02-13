//==============================================================================================//

#include "CUDABasedRasterization.h"

//==============================================================================================//

#define CLOCKS_PER_SEC ((clock_t)1000) 

//==============================================================================================//

CUDABasedRasterization::CUDABasedRasterization(std::vector<int>faces, std::vector<float>textureCoordinates, int numberOfVertices, std::vector<float>extrinsics, std::vector<float>intrinsics, int frameResolutionU, int frameResolutionV)
{
	//faces
	if(faces.size() % 3 == 0)
	{
		input.F = (faces.size() / 3);
		cutilSafeCall(cudaMalloc(&input.d_facesVertex, sizeof(int3) * input.F));
		cutilSafeCall(cudaMemcpy(input.d_facesVertex, faces.data(), sizeof(int3)*input.F, cudaMemcpyHostToDevice));

		// Get the vertexFaces, vertexFacesId
		std::vector<int> vertexFaces, vertexFacesId;
		getVertexFaces(numberOfVertices, faces, vertexFaces, vertexFacesId);
		cutilSafeCall(cudaMalloc(&input.d_vertexFaces, sizeof(int) * vertexFaces.size()));
		cutilSafeCall(cudaMemcpy(input.d_vertexFaces, vertexFaces.data(), sizeof(int)*vertexFaces.size(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMalloc(&input.d_vertexFacesId, sizeof(int) * vertexFacesId.size()));
		cutilSafeCall(cudaMemcpy(input.d_vertexFacesId, vertexFacesId.data(), sizeof(int)*vertexFacesId.size(), cudaMemcpyHostToDevice));
	}
	else
	{
		std::cout << "No triangular faces!" << std::endl;
	}

	//texture coordinates
	if (textureCoordinates.size() % 6 == 0)
	{
		cutilSafeCall(cudaMalloc(&input.d_textureCoordinates, sizeof(float) * 6 * input.F));
		cutilSafeCall(cudaMemcpy(input.d_textureCoordinates, textureCoordinates.data(), sizeof(float)*input.F * 6, cudaMemcpyHostToDevice));
	}
	else
	{
		std::cout << "Texture coordinates have wrong dimensionality!" << std::endl;
	}
	
	//camera parameters
	if (extrinsics.size() % 12 == 0 && intrinsics.size() % 9 == 0)
	{
		input.numberOfCameras = extrinsics.size()/12;
		cutilSafeCall(cudaMalloc(&input.d_cameraExtrinsics, sizeof(float4)*input.numberOfCameras * 3));
		cutilSafeCall(cudaMalloc(&input.d_cameraIntrinsics, sizeof(float3)*input.numberOfCameras * 3));
		cutilSafeCall(cudaMemcpy(input.d_cameraExtrinsics, extrinsics.data(), sizeof(float)*input.numberOfCameras * 3 * 4, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(input.d_cameraIntrinsics, intrinsics.data(), sizeof(float)*input.numberOfCameras * 3 * 3, cudaMemcpyHostToDevice));
	}
	else
	{
		std::cout << "Camera extrinsics or intrinsics coordinates have wrong dimensionality!" << std::endl;
		std::cout << "Extrinsics have dimension " << extrinsics.size() << std::endl;
		std::cout << "Intrinsics have dimension " << intrinsics.size() << std::endl;
	}

	input.w = frameResolutionU;
	input.h = frameResolutionV;

	//misc
	input.N = numberOfVertices;
	cutilSafeCall(cudaMalloc(&input.d_BBoxes,				sizeof(int4)   *	input.F*input.numberOfCameras));
	cutilSafeCall(cudaMalloc(&input.d_projectedVertices,	sizeof(float3) *	numberOfVertices * input.numberOfCameras));
	cutilSafeCall(cudaMalloc(&input.d_faceNormals,			sizeof(float3) *	input.F * input.numberOfCameras));
	cutilSafeCall(cudaMalloc(&input.d_vertexNormals,		sizeof(float3) *	input.N * input.numberOfCameras));
}

//==============================================================================================//

CUDABasedRasterization::~CUDABasedRasterization()
{
	cutilSafeCall(cudaFree(input.d_BBoxes));
	cutilSafeCall(cudaFree(input.d_projectedVertices));
	cutilSafeCall(cudaFree(input.d_cameraExtrinsics));
	cutilSafeCall(cudaFree(input.d_cameraIntrinsics));
	cutilSafeCall(cudaFree(input.d_textureCoordinates));
	cutilSafeCall(cudaFree(input.d_facesVertex));
}

//==============================================================================================//

void CUDABasedRasterization::getVertexFaces(int numberOfVertices, std::vector<int> faces, std::vector<int> &vertexFaces, std::vector<int> &vertexFacesId)
{
	int vertexId;
	int faceId;
	int startId;
	int numFacesPerVertex;
	
	for (int i = 0; i<numberOfVertices; i++) 
	{
		vertexId = i;
		startId = vertexFaces.size();
		
		for (int j = 0; j<faces.size(); j += 3)
		{
			faceId = int(j / 3);
			if (vertexId == faces[j] || vertexId == faces[j + 1] || vertexId == faces[j + 2])
			{
				vertexFaces.push_back(faceId);
			}
		}
		numFacesPerVertex = vertexFaces.size() - startId;
		if (numFacesPerVertex>0)
		{
			vertexFacesId.push_back(startId);
			vertexFacesId.push_back(numFacesPerVertex);
		}
		else
			std::cout << "WARNING:: --------- no faces for vertex " << vertexId << " --------- " << std::endl;
	}
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
