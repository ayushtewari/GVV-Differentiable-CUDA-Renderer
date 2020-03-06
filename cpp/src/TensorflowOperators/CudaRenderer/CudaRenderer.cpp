#include "CudaRenderer.h"

//==============================================================================================//

REGISTER_OP("CudaRendererGpu")

.Input("vertex_pos: float")
.Input("vertex_color: float")
.Input("texture: float")
.Input("sh_coeff: float")

.Output("barycentric_buffer: float")
.Output("face_buffer: int32")
.Output("depth_buffer: int32")

.Output("render_buffer: float")

.Output("vertex_normal: float")

.Attr("faces: list(int)")
.Attr("texture_coordinates: list(float)")
.Attr("number_of_vertices: int")
.Attr("extrinsics: list(float)")
.Attr("intrinsics: list(float)")
.Attr("render_resolution_u: int = 512")
.Attr("render_resolution_v: int = 512")
.Attr("render_mode: string");

//==============================================================================================//

CudaRenderer::CudaRenderer(OpKernelConstruction* context)
	: 
	OpKernel(context) 
{
	std::vector<int> faces;
	OP_REQUIRES_OK(context, context->GetAttr("faces", &faces));

	std::vector<float> textureCoordinates;
	OP_REQUIRES_OK(context, context->GetAttr("texture_coordinates", &textureCoordinates));

	OP_REQUIRES_OK(context, context->GetAttr("number_of_vertices", &numberOfPoints));
	OP_REQUIRES(context, numberOfPoints > 0, errors::InvalidArgument("number_of_vertices not set!", numberOfPoints));

	std::vector<float> extrinsics;
	OP_REQUIRES_OK(context, context->GetAttr("extrinsics", &extrinsics));
	if (extrinsics.size() % 12 == 0)
		numberOfCameras = extrinsics.size() / 12;
	else
		std::cout << "Camera extrinsics have wrong dimensionality!" << std::endl;

	std::vector<float> intrinsics;
	OP_REQUIRES_OK(context, context->GetAttr("intrinsics", &intrinsics));

	OP_REQUIRES_OK(context, context->GetAttr("render_resolution_u", &renderResolutionU));
	OP_REQUIRES(context, renderResolutionU > 0, errors::InvalidArgument("render_resolution_u not set!", renderResolutionU));

	OP_REQUIRES_OK(context, context->GetAttr("render_resolution_v", &renderResolutionV));
	OP_REQUIRES(context, renderResolutionV > 0, errors::InvalidArgument("render_resolution_v not set!", renderResolutionV));

	OP_REQUIRES_OK(context, context->GetAttr("render_mode", &renderMode));
	if (renderMode != "vertexColor" && renderMode != "textured")
	{
		std::cout << "INVALID RENDER MODE" << std::endl;
		return;
	}

	//---CONSOLE OUTPUT---

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	std::cout << "OPERATOR: CudaRenderer" << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	cudaBasedRasterization = new CUDABasedRasterization(faces, textureCoordinates, numberOfPoints, extrinsics, intrinsics, renderResolutionU, renderResolutionV, renderMode);
}

//==============================================================================================//

void CudaRenderer::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the 3D vertex position
	const Tensor& inputTensorVertexPos = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorVertexPosFlat = inputTensorVertexPos.flat_inner_dims<float, 1>();
	d_inputVertexPos = inputTensorVertexPosFlat.data();

	//[1]
	//Grab the vertex color
	const Tensor& inputTensorVertexColor = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorVertexColorFlat = inputTensorVertexColor.flat_inner_dims<float, 1>();
	d_inputVertexColor = inputTensorVertexColorFlat.data();

	//[2]
	//Grab the texture
	const Tensor& inputTensorTexture = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorTextureFlat = inputTensorTexture.flat_inner_dims<float, 1>();
	d_inputTexture = inputTensorTextureFlat.data();

	//[3]
	//Grab the sh coeffs 
	const Tensor& inputTensorSHCoeff = context->input(3);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorSHCoeffFlat = inputTensorSHCoeff.flat_inner_dims<float, 1>();
	d_inputSHCoeff = inputTensorSHCoeffFlat.data();

	//---MISC---

	numberOfBatches      = inputTensorVertexPos.dim_size(0); 
	textureResolutionV	 = inputTensorTexture.dim_size(1);
	textureResolutionU   = inputTensorTexture.dim_size(2);

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

	std::vector<tensorflow::int64> vertexDim;
	vertexDim.push_back(numberOfBatches);
	vertexDim.push_back(numberOfCameras);
	vertexDim.push_back(numberOfPoints);
	tensorflow::gtl::ArraySlice<tensorflow::int64> vertexDimSize(vertexDim);

	std::vector<tensorflow::int64> vertexNormalDim;
	vertexNormalDim.push_back(numberOfBatches);
	vertexNormalDim.push_back(numberOfCameras);
	vertexNormalDim.push_back(numberOfPoints);
	vertexNormalDim.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> vertexNormalDimSize(vertexNormalDim);

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
	//vertex normal
	tensorflow::Tensor* outputTensorVertexNormal;
	OP_REQUIRES_OK(context, context->allocate_output(4, tensorflow::TensorShape(vertexNormalDimSize), &outputTensorVertexNormal));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorVertexNormalFlat = outputTensorVertexNormal->flat<float>();
	d_outputVertexNormal = outputTensorVertexNormalFlat.data();
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
			cudaBasedRasterization->setTextureWidth(textureResolutionU);
			cudaBasedRasterization->setTextureHeight(textureResolutionV);
			cudaBasedRasterization->set_D_vertices(			(float3*)   d_inputVertexPos						+ b * numberOfPoints * 3);
			cudaBasedRasterization->set_D_vertexColors(		(float3*)	d_inputVertexColor						+ b * numberOfPoints * 3);
			cudaBasedRasterization->set_D_textureMap(					d_inputTexture							+ b * textureResolutionV * textureResolutionU * 3);
			cudaBasedRasterization->set_D_shCoeff(						d_inputSHCoeff							+ b * numberOfCameras * 27);

			//set output
			cudaBasedRasterization->set_D_barycentricCoordinatesBuffer(	d_outputBarycentricCoordinatesBuffer	+ b * numberOfCameras * renderResolutionV * renderResolutionU * 3);
			cudaBasedRasterization->set_D_faceIDBuffer(					d_outputFaceIDBuffer					+ b * numberOfCameras * renderResolutionV * renderResolutionU * 4);
			cudaBasedRasterization->set_D_depthBuffer(					d_outputDepthBuffer						+ b * numberOfCameras * renderResolutionV * renderResolutionU);
			cudaBasedRasterization->set_D_renderBuffer(					d_outputRenderBuffer					+ b * numberOfCameras * renderResolutionV * renderResolutionU * 3);

			cudaBasedRasterization->set_D_vertexNormal(		(float3*)	d_outputVertexNormal					+ b * numberOfCameras * numberOfPoints * 3);

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
