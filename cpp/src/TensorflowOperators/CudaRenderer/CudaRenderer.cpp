#include "CudaRenderer.h"

//==============================================================================================//

REGISTER_OP("CudaRendererGpu")
.Input("points_global_space: float")

.Output("barycentric_buffer: float")
.Output("face_buffer: int32")
.Output("depth_buffer: int32")
.Output("render_buffer: float")
.Output("vertex_color_buffer: float")
.Attr("faces: list(int)")
.Attr("render_resolution_u: int = 512")
.Attr("render_resolution_v: int = 512")
.Attr("camera_file_path_boundary_check: string = 'None'")
.Attr("mesh_file_path_boundary_check: string = 'None'");

//==============================================================================================//

CudaRenderer::CudaRenderer(OpKernelConstruction* context)
	: 
	OpKernel(context) 
{
	std::vector<int> faces;

	OP_REQUIRES_OK(context, context->GetAttr("faces", &faces));

	OP_REQUIRES_OK(context, context->GetAttr("camera_file_path_boundary_check", &cameraFilePath));
	OP_REQUIRES(context,cameraFilePath != std::string("None"),errors::InvalidArgument("camera_file_path_boundary_check not set!",cameraFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("mesh_file_path_boundary_check", &meshFilePath));
	OP_REQUIRES(context,meshFilePath != std::string("None"),errors::InvalidArgument("mesh_file_path_boundary_check not set!",meshFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("render_resolution_u", &renderResolutionU));
	OP_REQUIRES(context, renderResolutionU > 0, errors::InvalidArgument("render_resolution_u not set!", renderResolutionU));

	OP_REQUIRES_OK(context, context->GetAttr("render_resolution_v", &renderResolutionV));
	OP_REQUIRES(context, renderResolutionV > 0, errors::InvalidArgument("render_resolution_v not set!", renderResolutionV));

	cameras = new camera_container();
	cameras->loadCameras(cameraFilePath.c_str());

	mesh = new trimesh(meshFilePath.c_str());
	mesh->setupViewDependedGPUMemory(cameras->getNrCameras());

	numberOfCameras = cameras->getNrCameras();

	//---CONSOLE OUTPUT---

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	std::cout << "OPERATOR: ProjectedMeshBoundaryGpu" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) Points Global Space dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(0) Points Global Space dimension " << 0 << " size: " << "batch size" << std::endl;
	std::cout << "	" << "Input(0) Points Global Space dimension " << 1 << " size: " << std::to_string(mesh->N) << std::endl;
	std::cout << "	" << "Input(0) Points Global Space dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) Is Boundary dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(0) Is Boundary dimension " << 0 << " size: " << "batch size" << std::endl;
	std::cout << "	" << "Ouput(0) Is Boundary dimension " << 1 << " size: " << std::to_string(cameras->getNrCameras()) << std::endl;
	std::cout << "	" << "Ouput(0) Is Boundary dimension " << 2 << " size: " << std::to_string(mesh->N) << std::endl;

	std::cout << std::endl;

	std::cout << "Attr(0) Character File Path: " << cameraFilePath << std::endl;

	std::cout << "Attr(1) Mesh File Path: " << meshFilePath << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	cudaBasedRasterization = new CUDABasedRasterization(mesh, cameras, faces);
}

//==============================================================================================//

void CudaRenderer::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the 3D points global space
	const Tensor& inputTensorPointsGlobalSpace = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorPointsGlobalSpaceFlat = inputTensorPointsGlobalSpace.flat_inner_dims<float, 1>();
	d_inputDataPointerPointsGlobalSpace = inputTensorPointsGlobalSpaceFlat.data();

	//---MISC---

	numberOfBatches = inputTensorPointsGlobalSpace.dim_size(0); // aka number of meshes/skeletons
	numberOfPoints = inputTensorPointsGlobalSpace.dim_size(1); // aka number of cameras

	//---OUTPUT---

	//determine the output dimensions
	std::vector<tensorflow::int64> channel1Dim;
	channel1Dim.push_back(numberOfBatches);
	channel1Dim.push_back(numberOfCameras);
	channel1Dim.push_back(renderResolutionV);
	channel1Dim.push_back(renderResolutionU);
	tensorflow::gtl::ArraySlice<tensorflow::int64> channel1DimSize(channel1Dim);

	std::vector<tensorflow::int64> channel3Dim;
	channel3Dim.push_back(numberOfBatches);
	channel3Dim.push_back(numberOfCameras);
	channel3Dim.push_back(renderResolutionV);
	channel3Dim.push_back(renderResolutionU);
	channel3Dim.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> channel3DimSize(channel3Dim);

	std::vector<tensorflow::int64> channel4Dim;
	channel4Dim.push_back(numberOfBatches);
	channel4Dim.push_back(numberOfCameras);
	channel4Dim.push_back(renderResolutionV);
	channel4Dim.push_back(renderResolutionU);
	channel4Dim.push_back(4);
	tensorflow::gtl::ArraySlice<tensorflow::int64> channel4DimSize(channel4Dim);

	//[0]
	//barycentric
	tensorflow::Tensor* outputTensorBarycentric;
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(channel3DimSize), &outputTensorBarycentric));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorBarycentricFlat = outputTensorBarycentric->flat<float>();
	d_outputBarycentricCoordinatesBuffer = outputTensorBarycentricFlat.data();

	//[1]
	//face
	tensorflow::Tensor* outputTensorFace;
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(channel4DimSize), &outputTensorFace));
	Eigen::TensorMap<Eigen::Tensor<int, 1, 1, Eigen::DenseIndex>, 16> outputTensorFaceFlat = outputTensorFace->flat<int>();
	d_outputFaceIDBuffer = outputTensorFaceFlat.data();

	//[2]
	//depth
	tensorflow::Tensor* outputTensorDepth;
	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(channel1DimSize), &outputTensorDepth));
	Eigen::TensorMap<Eigen::Tensor<int, 1, 1, Eigen::DenseIndex>, 16> outputTensorDepthFlat = outputTensorDepth->flat<int>();
	d_outputDepthBuffer = outputTensorDepthFlat.data();

	//[3]
	//render
	tensorflow::Tensor* outputTensorRender;
	OP_REQUIRES_OK(context, context->allocate_output(3, tensorflow::TensorShape(channel3DimSize), &outputTensorRender));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorRenderFlat = outputTensorRender->flat<float>();
	d_outputRenderBuffer = outputTensorRenderFlat.data();

	//[4]
	//render
	tensorflow::Tensor* outputTensorVertexColor;
	OP_REQUIRES_OK(context, context->allocate_output(4, tensorflow::TensorShape(channel3DimSize), &outputTensorVertexColor));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorVertexColorFlat = outputTensorVertexColor->flat<float>();
	d_outputVertexColorBuffer = outputTensorVertexColorFlat.data();
}

//==============================================================================================//

void CudaRenderer::Compute(OpKernelContext* context)
{
	try
	{
		//setup the input and output pointers of the tensor because they change from compute to compute call
		setupInputOutputTensorPointers(context);

		for (int b = 0; b < numberOfBatches; b++)
		{
			//set input 
			cudaBasedRasterization->set_D_vertices( (float3*)d_inputDataPointerPointsGlobalSpace + b * numberOfPoints);
			
			//set output
			cudaBasedRasterization->set_D_barycentricCoordinatesBuffer(	d_outputBarycentricCoordinatesBuffer	+ b * numberOfCameras * renderResolutionV * renderResolutionU * 3);
			cudaBasedRasterization->set_D_faceIDBuffer(					d_outputFaceIDBuffer					+ b * numberOfCameras * renderResolutionV * renderResolutionU * 4);
			cudaBasedRasterization->set_D_depthBuffer(					d_outputDepthBuffer						+ b * numberOfCameras * renderResolutionV * renderResolutionU);
			cudaBasedRasterization->set_D_renderBuffer(					d_outputRenderBuffer					+ b * numberOfCameras * renderResolutionV * renderResolutionU * 3);
			cudaBasedRasterization->set_D_vertexColorBuffer(			d_outputVertexColorBuffer				+ b * numberOfCameras * renderResolutionV * renderResolutionU * 3);

			//render
			cudaBasedRasterization->renderBuffers();
		}
	}
	catch (std::exception e)
	{
		std::cerr << "Compute projected mesh boundary error!" << std::endl;
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("CudaRendererGpu").Device(DEVICE_GPU), CudaRenderer);
