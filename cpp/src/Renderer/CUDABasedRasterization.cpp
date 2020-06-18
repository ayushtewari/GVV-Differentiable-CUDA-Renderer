//==============================================================================================//

#include "CUDABasedRasterization.h"

//==============================================================================================//

#define CLOCKS_PER_SEC ((clock_t)1000) 

//==============================================================================================//

CUDABasedRasterization::CUDABasedRasterization(
	std::vector<int>faces, 
	std::vector<float>textureCoordinates, 
	int numberOfVertices, 
	std::vector<float>extrinsics, 
	std::vector<float>intrinsics, 
	int frameResolutionU, 
	int frameResolutionV, 
	std::string albedoMode, 
	std::string shadingMode)
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

		cutilSafeCall(cudaMalloc(&input.d_inverseExtrinsics,		sizeof(float4)*input.numberOfCameras * 4));
		cutilSafeCall(cudaMalloc(&input.d_inverseProjection,		sizeof(float4)*input.numberOfCameras * 4));
	
		for (int idc = 0; idc < input.numberOfCameras; idc++)
		{
			float4x4 h_intrinsics;
			h_intrinsics.setIdentity();
			h_intrinsics(0, 0) = intrinsics.data()[9 * idc + 0];
			h_intrinsics(0, 1) = intrinsics.data()[9 * idc + 1];
			h_intrinsics(0, 2) = intrinsics.data()[9 * idc + 2];
			h_intrinsics(0, 3) = 0.f;

			h_intrinsics(1, 0) = intrinsics.data()[9 * idc + 3];
			h_intrinsics(1, 1) = intrinsics.data()[9 * idc + 4];
			h_intrinsics(1, 2) = intrinsics.data()[9 * idc + 5];
			h_intrinsics(1, 3) = 0.f;

			h_intrinsics(2, 0) = intrinsics.data()[9 * idc + 6];
			h_intrinsics(2, 1) = intrinsics.data()[9 * idc + 7];
			h_intrinsics(2, 2) = intrinsics.data()[9 * idc + 8];
			h_intrinsics(2, 3) = 0.f;


			float4x4 h_extrinsics;
			h_extrinsics.setIdentity();
			h_extrinsics(0, 0) = extrinsics.data()[12 * idc + 0];
			h_extrinsics(0, 1) = extrinsics.data()[12 * idc + 1];
			h_extrinsics(0, 2) = extrinsics.data()[12 * idc + 2];
			h_extrinsics(0, 3) = extrinsics.data()[12 * idc + 3];

			h_extrinsics(1, 0) = extrinsics.data()[12 * idc + 4];
			h_extrinsics(1, 1) = extrinsics.data()[12 * idc + 5];
			h_extrinsics(1, 2) = extrinsics.data()[12 * idc + 6];
			h_extrinsics(1, 3) = extrinsics.data()[12 * idc + 7];

			h_extrinsics(2, 0) = extrinsics.data()[12 * idc + 8];
			h_extrinsics(2, 1) = extrinsics.data()[12 * idc + 9];
			h_extrinsics(2, 2) = extrinsics.data()[12 * idc + 10];
			h_extrinsics(2, 3) = extrinsics.data()[12 * idc + 11];

			float4x4 h_inExtrinsics = h_extrinsics.getInverse();
			float4x4 h_invProjection = (h_intrinsics * h_extrinsics).getInverse();
			cutilSafeCall(cudaMemcpy(input.d_inverseExtrinsics       + idc * 4,	(float4*)&h_inExtrinsics(0, 0),			sizeof(float4) * 4, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(input.d_inverseProjection		 + idc * 4, (float4*)&h_invProjection(0, 0),		sizeof(float4) * 4, cudaMemcpyHostToDevice));
		}	
	}
	else
	{
		std::cout << "Camera extrinsics or intrinsics coordinates have wrong dimensionality!" << std::endl;
		std::cout << "Extrinsics have dimension " << extrinsics.size() << std::endl;
		std::cout << "Intrinsics have dimension " << intrinsics.size() << std::endl;
	}

	input.w = frameResolutionU;
	input.h = frameResolutionV;

	//render mode
	if (albedoMode == "vertexColor")
	{
		input.albedoMode = AlbedoMode::VertexColor;
	}
	else if (albedoMode == "textured")
	{
		input.albedoMode = AlbedoMode::Textured;
	}
	else if (albedoMode == "normal")
	{
		input.albedoMode = AlbedoMode::Normal;
	}
	else if (albedoMode == "lighting")
	{
		input.albedoMode = AlbedoMode::Lighting;
	}
	else if (albedoMode == "foregroundMask")
	{
		input.albedoMode = AlbedoMode::ForegroundMask;
	}
	

	//shading mode
	if (shadingMode == "shaded")
	{
		input.shadingMode = ShadingMode::Shaded;
	}
	else if (shadingMode == "shadeless")
	{
		input.shadingMode = ShadingMode::Shadeless;
	}

	//misc
	input.N = numberOfVertices;
	cutilSafeCall(cudaMalloc(&input.d_BBoxes,				sizeof(int4)   *	input.F*input.numberOfCameras));
	cutilSafeCall(cudaMalloc(&input.d_projectedVertices,	sizeof(float3) *	numberOfVertices * input.numberOfCameras));
	cutilSafeCall(cudaMalloc(&input.d_faceNormal,			sizeof(float3) *	input.F * input.numberOfCameras));

	cutilSafeCall(cudaMalloc(&input.d_depthBuffer, sizeof(int) * input.numberOfCameras * input.h * input.w ));

	textureMapFaceIdSet = false;
	texCoords = textureCoordinates;
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
	cutilSafeCall(cudaFree(input.d_vertexFaces));
	cutilSafeCall(cudaFree(input.d_vertexFacesId));
	cutilSafeCall(cudaFree(input.d_faceNormal));
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

bool rayTriangleIntersectHost(float3 orig, float3 dir, float3 v0, float3 v1, float3 v2, float &t, float &a, float &b)
{
	//just to make it numerically more stable
	v0 = v0 / 1000.f;
	v1 = v1 / 1000.f;
	v2 = v2 / 1000.f;
	orig = orig / 1000.f;

	// compute plane's normal
	float3  v0v1 = v1 - v0;
	float3  v0v2 = v2 - v0;

	// no need to normalize
	float3  N = cross(v0v1, v0v2); // N 

								   /////////////////////////////
								   // Step 1: finding P
								   /////////////////////////////

								   // check if ray and plane are parallel ?
	float NdotRayDirection = dot(dir, N);
	if (fabs(NdotRayDirection) < 0.0000001f) // almost 0 
	{
		return false; // they are parallel so they don't intersect ! 
	}
	// compute d parameter using equation 2
	float d = dot(N, v0);

	// compute t (equation 3)
	t = (dot(v0, N) - dot(orig, N)) / NdotRayDirection;
	// check if the triangle is in behind the ray
	if (t < 0)
	{
		return false; // the triangle is behind 
	}
	// compute the intersection point using equation 1
	float3 P = orig + t * dir;

	/////////////////////////////
	// Step 2: inside-outside test
	/////////////////////////////

	float3 C; // vector perpendicular to triangle's plane 

			  // edge 0
	float3 edge0 = v1 - v0;
	float3 vp0 = P - v0;
	C = cross(edge0, vp0);
	if (dot(N, C) < 0)
	{
		return false;
	}
	// edge 1
	float3 edge1 = v2 - v1;
	float3 vp1 = P - v1;
	C = cross(edge1, vp1);
	if ((a = dot(N, C)) < 0)
	{
		return false;
	}
	// edge 2
	float3 edge2 = v0 - v2;
	float3 vp2 = P - v2;
	C = cross(edge2, vp2);

	if ((b = dot(N, C)) < 0)
	{
		return false;
	}

	float denom = dot(N, N);
	a /= denom;
	b /= denom;

	return true; // this ray hits the triangle 
}

//==============================================================================================//

void CUDABasedRasterization::renderBuffers()
{
	//init the texture map face ids 
	if (!textureMapFaceIdSet)
	{
		//texture map ids 
		float4* h_textureMapFaceIds = new float4[input.texHeight * input.texWidth];
		cutilSafeCall(cudaMalloc(&input.d_textureMapIds, sizeof(float4) *	input.texHeight * input.texWidth));

		for (int x = 0; x < input.texWidth; x++)
		{
			for (int y = 0; y < input.texHeight; y++)
			{
				//init pixel
				h_textureMapFaceIds[y * input.texWidth + x] = make_float4(0, 0, 0, 0);

				//pixel ray
				float3 d = make_float3(0.f, 0.f, -1.f);
				float3 o = make_float3(x + 0.5f, y + 0.5f, 1.f);

				//check if it is inside a triangle
				for (int f = 0; f < input.F; f++)
				{
					float3 texCoord0 = make_float3(input.texWidth * texCoords[f * 3 * 2 + 0 * 2 + 0], input.texHeight * (1.f - texCoords[f * 3 * 2 + 0 * 2 + 1]), 0.f);
					float3 texCoord1 = make_float3(input.texWidth * texCoords[f * 3 * 2 + 1 * 2 + 0], input.texHeight * (1.f - texCoords[f * 3 * 2 + 1 * 2 + 1]), 0.f);
					float3 texCoord2 = make_float3(input.texWidth * texCoords[f * 3 * 2 + 2 * 2 + 0], input.texHeight * (1.f - texCoords[f * 3 * 2 + 2 * 2 + 1]), 0.f);
					
					float a, b,c, t;

					bool intersect = rayTriangleIntersectHost(o, d, texCoord0, texCoord1, texCoord2, t, a, b);

					if (!intersect)
						a = b = c = -1.f;
					else
						c = 1.f - a - b;

					if (a != -1.f && b != -1.f && c != -1.f)
					{	
						std::cout << "Pixel: " << x << " | " << y << " : " << f << " " << std::to_string(a) << " " << std::to_string(b) << " " << std::to_string(c) << std::endl;
						h_textureMapFaceIds[y * input.texWidth + x] = make_float4(f, a, b, c);
					}
				}
			}
		}

		cutilSafeCall(cudaMemcpy(input.d_textureMapIds, h_textureMapFaceIds, sizeof(float4) *	input.texHeight * input.texWidth, cudaMemcpyHostToDevice));
		textureMapFaceIdSet = true;
	}

	renderBuffersGPU(input);
}
