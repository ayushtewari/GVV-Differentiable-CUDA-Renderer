
//==============================================================================================//
// Classname:
//      CUDABasedRasterizationInput
//
//==============================================================================================//
// Description:
//      Data structure for the Cuda based rasterization
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h> 

//==============================================================================================//

#define THREADS_PER_BLOCK_CUDABASEDRASTERIZER 256

//==============================================================================================//

struct CUDABasedRasterizationInput
{
	//////////////////////////
	//CONSTANT INPUTS
	//////////////////////////

	//camera and frame
	int					numberOfCameras;						//number of cameras
	float4*				d_cameraExtrinsics;						//camera extrinsics
	float3*				d_cameraIntrinsics;						//camera intrinsics
	int					w;										//frame width
	int					h;										//frame height

	//geometry
	int					F;										//number of faces
	int					N;										//number of vertices
	int3*				d_facesVertex;							//part of face data structure

	//texture 
	int					texWidth;
	int					texHeight;
	float*				d_textureCoordinates;

	//////////////////////////
	//INPUTS
	//////////////////////////

	float3*				d_vertices;								//vertex positions
	uchar3*				d_vertexColor;

	//texture
	float*				d_textureMap;

	//////////////////////////
	//OUTPUT 
	//////////////////////////

	//render buffers
	int*				d_faceIDBuffer;							//face ID per pixel per view and the ids of the 3 vertices
	int*				d_depthBuffer;							//depth value per pixel per view
	float*				d_barycentricCoordinatesBuffer;			//barycentric coordinates per pixel per view
	float*				d_renderBuffer;
	float*				d_vertexColorBuffer;

	//visibility and boundary per vertex
	bool*				d_visibilities;							//is visible flag (per vertex per view)
	bool*				d_boundaries;							//is boundary flag (per vertex per view)

	//////////////////////////
	//STATES 
	//////////////////////////
																//misc
	int4*				d_BBoxes;								//bbox for each triangle
	float3*				d_projectedVertices;					//vertex position on image with depth after projection
};