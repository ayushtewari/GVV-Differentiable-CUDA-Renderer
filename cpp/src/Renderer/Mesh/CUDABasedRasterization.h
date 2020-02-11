//==============================================================================================//

#pragma once

//==============================================================================================//

#include <set>
#include "time.h"
#include <iostream>
#include "CUDABasedRasterizationInput.h"
#include "../Camera/camera_container.h"
#include "../Mesh/trimesh.h"

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

		CUDABasedRasterization(trimesh* mesh, camera_container* cm,
			std::vector<int>faces);
		~CUDABasedRasterization();

		void renderBuffers();
		void checkVisibility(bool checkBoundary);

		void copyDepthBufferGPU2CPU();
		void copyBarycentricBufferGPU2CPU();
		void copyFaceIdBufferGPU2CPU();
		void copyRenderBufferGPU2CPU();
		void copyVertexColorBufferGPU2CPU();

		//=================================================//
		//=================================================//

		//getter
		
		camera_container*						getCameras()								{ return cameras; };
		trimesh*								getMesh()									{ return mesh; };

		//getter for geometry
		inline int								getNumberOfFaces()							{ return input.F;};
		inline int								getNumberOfVertices()						{ return input.N; };
		inline int3*							get_D_facesVertex()							{ return input.d_facesVertex; };
		inline float3*							get_D_vertices()							{ return input.d_vertices; };
		inline bool*							get_D_visibilities()						{ return input.d_visibilities; };
		inline bool*							get_D_boundaries()							{ return input.d_boundaries; };
		inline uchar3*							get_D_vertexColor()							{ return input.d_vertexColor; };

		//getter for texture
		inline float*							get_D_textureCoordinates()					{ return input.d_textureCoordinates; };
		inline float*							get_D_textureMap()							{ return input.d_textureMap; };
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
		inline void							setCameras(camera_container* inputCameras)						{ cameras=inputCameras; }
		inline void							setMesh(trimesh* inputMesh)										{ mesh=inputMesh; }
		inline void							set_D_vertices(float3* d_inputVertices)							{ input.d_vertices = d_inputVertices; };
		inline void							set_D_boundaries(bool* d_inputBoundaries)						{ input.d_boundaries = d_inputBoundaries; };

		inline void							set_D_faceIDBuffer(int* newFaceBuffer)							{ input.d_faceIDBuffer = newFaceBuffer; };
		inline void							set_D_depthBuffer(int* newDepthBuffer)							{ input.d_depthBuffer = newDepthBuffer; };
		inline void							set_D_barycentricCoordinatesBuffer(float* newBarycentricBuffer) { input.d_barycentricCoordinatesBuffer = newBarycentricBuffer; };
		inline void							set_D_renderBuffer(float* newRenderBuffer)						{ input.d_renderBuffer = newRenderBuffer; };
		inline void							set_D_vertexColorBuffer(float* newVertexColorbuffer)			{ input.d_vertexColorBuffer = newVertexColorbuffer; };

	//variables

	private:

		//device memory
		CUDABasedRasterizationInput input;

		//host memory
		float*						h_barycentricCoordinatesBuffer;
		int4*						h_faceIDBuffer;				
		int*						h_depthBuffer;				
		float*						h_renderBuffer;
		float*						h_vertexColorBuffer;

		camera_container*           cameras;
		trimesh*                    mesh;
};

//==============================================================================================//

//#endif // SKELETONIZE_INTERFACE_H
