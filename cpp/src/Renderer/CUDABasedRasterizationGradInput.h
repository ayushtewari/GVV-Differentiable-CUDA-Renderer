
//==============================================================================================//
// Classname:
//      CUDABasedRasterizationGradInput
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

struct CUDABasedRasterizationGradInput
{
	//////////////////////////
	//CONSTANT INPUTS
	//////////////////////////

	//camera and frame
	int					numberOfCameras;						//number of cameras													//INIT IN CONSTRUCTOR
	float4*				d_cameraExtrinsics;						//camera extrinsics													//INIT IN CONSTRUCTOR
	float3*				d_cameraIntrinsics;						//camera intrinsics													//INIT IN CONSTRUCTOR			////
	int					w;										//frame width														//INIT IN CONSTRUCTOR
	int					h;										//frame height														//INIT IN CONSTRUCTOR

	//geometry
	int					F;										//number of faces													//INIT IN CONSTRUCTOR			////
	int					N;										//number of vertices												//INIT IN CONSTRUCTOR
	int3*				d_facesVertex;							//part of face data structure										//INIT IN CONSTRUCTOR

	//texture	
	float*				d_textureCoordinates;																						//INIT IN CONSTRUCTOR			////

	//////////////////////////
	//STATES 
	//////////////////////////

	//misc
	int*                d_vertexFaces;                          //list of neighbourhood faces for each vertex						//INIT IN CONSTRUCTOR
	int2*               d_vertexFacesId;                        //list of (index in d_vertexFaces, number of faces) for each vertex	//INIT IN CONSTRUCTOR

		
	//////////////////////////
	//INPUTS
	//////////////////////////
	
	float3*				d_vertexColorBufferGrad;

	float3*				d_vertices;								//vertex positions
	float3*				d_vertexColor;							//vertex color
	float3*				d_vertexNormal;							//vertex normals												
	//texture
	int					texWidth;								//dimension of texture																				///
	int					texHeight;								//dimension of texture																				///
	const float*		d_textureMap;							//texture map																						///
	const float*		d_shCoeff;								//shading coefficients

	//////////////////////////
	//OUTPUT 
	//////////////////////////

	//render buffers
	int4*				d_faceIDBuffer;							//face ID per pixel per view and the ids of the 3 vertices
	float3*				d_barycentricCoordinatesBuffer;			//barycentric coordinates per pixel per view


	//////////////////////////
	//Gradients
	//////////////////////////

	float3*				d_vertexPosGrad;
	float3*				d_vertexColorGrad;
	float*				d_shCoeffGrad;
};
