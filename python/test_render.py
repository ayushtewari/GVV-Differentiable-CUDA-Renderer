import data.test_mesh_tensor as test_mesh_tensor
import data.test_SH_tensor as test_SH_tensor
import CudaRenderer
import utils.CheckGPU as CheckGPU
import cv2 as cv
import utils.OBJReader as OBJReader
import utils.CameraReader as CameraReader


freeGPU = CheckGPU.get_free_gpu()

if freeGPU:

    testMesh3D = test_mesh_tensor.getGTMesh()
    testSHCoeff = test_SH_tensor.getSHCoeff()

    objreader = OBJReader.OBJReader('data/magdalena.obj')
    cameraReader = CameraReader.CameraReader('data/cameras.calibration')

    renderer = CudaRenderer.CudaRendererGpu(
                                            faces_attr                  = objreader.facesVertexId,
                                            texCoords_attr              = objreader.textureCoordinates,
                                            numberOfVertices_attr       = len(objreader.vertexCoordinates),
                                            extrinsics_attr             = cameraReader.extrinsics,
                                            intrinsics_attr             = cameraReader.intrinsics,
                                            renderResolutionU_attr      = 1024,
                                            renderResolutionV_attr      = 1024,

                                            vertexPos_input             = testMesh3D,
                                            vertexColor_input           = [objreader.vertexColors],
                                            texture_input               = [objreader.textureMap],
                                            shCoeff_input=testSHCoeff,

                                            nodeName                    = 'test')

    vertexColorBuffer = renderer.getRenderBuffer()[0][0].numpy() * 255.0

    vertexColorBuffer = cv.cvtColor(vertexColorBuffer, cv.COLOR_BGR2RGB)
    cv.imwrite('result/new.png',vertexColorBuffer)