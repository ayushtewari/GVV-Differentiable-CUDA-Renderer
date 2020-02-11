import data.test_mesh_tensor as test_mesh_tensor
import CudaRenderer
import utils.CheckGPU as CheckGPU
import cv2 as cv
import utils.OBJReader as OBJReader
freeGPU = CheckGPU.get_free_gpu()

if freeGPU:
    testMesh3D = test_mesh_tensor.getGTMesh()

    objreader = OBJReader.OBJReader('data/magdalena.obj')

    renderer = CudaRenderer.CudaRendererGpu(
                     faces = objreader.facesVertexId,
                     cameraFilePath = 'data/cameras.calibration',
                     meshFilePath = 'data/magdalena.obj',
                     renderResolutionU = 1024,
                     renderResolutionV = 1024,
                     pointsGlobalSpace = testMesh3D,
                     nodeName='test')

    vertexColorBuffer = renderer.getVertexColorBuffer()[0][0].numpy()
    cv.imwrite('result/new.png',vertexColorBuffer)