//==============================================================================================//

#pragma once

//==============================================================================================//

#include <set>
#include "time.h"
#include <iostream>
#include "CUDABasedRasterizationInput.h"
#include <vector>
#include <cuda_runtime.h>
#include "cutil.h"
#include "cutil_inline_runtime.h"
#include "cutil_math.h"

//==============================================================================================//

extern "C" void renderBuffersGPU(CUDABasedRasterizationInput& input);
extern "C" void checkVisibilityGPU(CUDABasedRasterizationInput& input, bool checkBoundary);

//==============================================================================================//

class CUDABasedRasterization
{
	//functions

	public:

		//=================================================//
		//=================================================//

		CUDABasedRasterization(std::vector<int>faces, std::vector<float>textureCoordinates, int numberOfVertices, std::vector<float>extrinsics, std::vector<float>intrinsics, int frameResolutionU, int frameResolutionV);
		~CUDABasedRasterization();

		void renderBuffers();
		void checkVisibility(bool checkBoundary);

		//=================================================//
		//=================================================//

		//getter
		
		//getter for geometry
		inline int								getNumberOfFaces()							{ return input.F;};
		inline int								getNumberOfVertices()						{ return input.N; };
		inline int3*							get_D_facesVertex()							{ return input.d_facesVertex; };
		inline float3*							get_D_vertices()							{ return input.d_vertices; };
		inline bool*							get_D_visibilities()						{ return input.d_visibilities; };
		inline bool*							get_D_boundaries()							{ return input.d_boundaries; };
		inline float3*							get_D_vertexColor()							{ return input.d_vertexColor; };

		//getter for texture
		inline float*							get_D_textureCoordinates()					{ return input.d_textureCoordinates; };
		inline const float*						get_D_textureMap()							{ return input.d_textureMap; };
		inline int								getTextureWidth()							{ return input.texWidth; };
		inline int								getTextureHeight()							{ return input.texHeight; };

		//getter for misc
		inline int4*							get_D_BBoxes()								{ return input.d_BBoxes; };
		inline float3*							get_D_projectedVertices()					{ return input.d_projectedVertices; };
	
		//getter for camera and frame
		inline int								getNrCameras()								{ return input.numberOfCameras; };
		inline float4*							get_D_cameraExtrinsics()					{ return input.d_cameraExtrinsics; };
		inline float3*							get_D_cameraIntrinsics()					{ return input.d_cameraIntrinsics; };
		inline int								getFrameWidth()								{ return input.w; };
		inline int								getFrameHeight()							{ return input.h; };
	
		//getter for render buffers
		inline int*							    get_D_faceIDBuffer()						{ return input.d_faceIDBuffer; };
		inline int*								get_D_depthBuffer()							{ return input.d_depthBuffer; };
		inline float*							get_D_barycentricCoordinatesBuffer()		{ return input.d_barycentricCoordinatesBuffer; };
		inline float*							get_D_renderBuffer()						{ return input.d_renderBuffer; };
		inline float*							get_D_vertexColorBuffer()					{ return input.d_vertexColorBuffer; };

		//=================================================//
		//=================================================//

		//setter
		inline void							set_D_vertices(float3* d_inputVertices)							{ input.d_vertices = d_inputVertices; };
		inline void							set_D_vertexColors(float3* d_inputVertexColors)					{ input.d_vertexColor = d_inputVertexColors; };
		inline void							set_D_textureMap(const float* newTextureMap)					{ input.d_textureMap = newTextureMap; };
		inline void							setTextureWidth(int newTextureWidth)							{ input.texWidth = newTextureWidth; };
		inline void							setTextureHeight(int newTextureHeight)							{ input.texHeight = newTextureHeight; };


		inline void							set_D_faceIDBuffer(int* newFaceBuffer)							{ input.d_faceIDBuffer = newFaceBuffer; };
		inline void							set_D_depthBuffer(int* newDepthBuffer)							{ input.d_depthBuffer = newDepthBuffer; };
		inline void							set_D_barycentricCoordinatesBuffer(float* newBarycentricBuffer) { input.d_barycentricCoordinatesBuffer = newBarycentricBuffer; };
		inline void							set_D_renderBuffer(float* newRenderBuffer)						{ input.d_renderBuffer = newRenderBuffer; };
		inline void							set_D_vertexColorBuffer(float* newVertexColorbuffer)			{ input.d_vertexColorBuffer = newVertexColorbuffer; };

		inline void							set_D_boundaries(bool* d_inputBoundaries)						{ input.d_boundaries = d_inputBoundaries; };
		inline void							set_D_visibilities(bool* d_inputvisibilities)					{ input.d_visibilities = d_inputvisibilities; };


	//variables

	private:

		//device memory
		CUDABasedRasterizationInput input;
};

//==============================================================================================//

//#endif // SKELETONIZE_INTERFACE_H
