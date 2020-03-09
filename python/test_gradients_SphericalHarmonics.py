########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from sys import platform
import data.test_mesh_tensor as test_mesh_tensor
import data.test_SH_tensor as test_SH_tensor
import CudaRenderer
import utils.CheckGPU as CheckGPU
import cv2 as cv
import utils.OBJReader as OBJReader
import utils.CameraReader as CameraReader

########################################################################################################################
# Load custom operators
########################################################################################################################

RENDER_OPERATORS_PATH = ""
if platform == "linux" or platform == "linux2":
    RENDER_OPERATORS_PATH = "../cpp/binaries/Linux/Release/libCustomTensorFlowOperators.so"
elif platform == "win32" or platform == "win64":
    RENDER_OPERATORS_PATH = "../cpp/binaries/Win64/Release/CustomTensorFlowOperators.dll"
customOperators = tf.load_op_library(RENDER_OPERATORS_PATH)

########################################################################################################################
# CudaRendererGpu class
########################################################################################################################

testMesh3D = test_mesh_tensor.getGTMesh()
testSHCoeff = test_SH_tensor.getSHCoeff(14)
objreader = OBJReader.OBJReader('data/magdalena.obj')
cameraReader = CameraReader.CameraReader('data/cameras.calibration')

########################################################################################################################
# test spherical harmonics lighting
########################################################################################################################

def test_SHC_gradient():

    VertexPosConst=tf.constant(testMesh3D,dtype=tf.float32)
    VertexColorConst=tf.constant([objreader.vertexColors],dtype=tf.float32)
    VertexTextureConst=tf.constant([objreader.textureMap],dtype=tf.float32)
    SHCConst=tf.constant(testSHCoeff,dtype=tf.float32)

    target = CudaRenderer.CudaRendererGpu(
                                            faces_attr=objreader.facesVertexId,
                                            texCoords_attr=objreader.textureCoordinates,
                                            numberOfVertices_attr=len(objreader.vertexCoordinates),
                                            extrinsics_attr=cameraReader.extrinsics,
                                            intrinsics_attr=cameraReader.intrinsics,
                                            renderResolutionU_attr=1024,
                                            renderResolutionV_attr=1024,
                                            renderMode_attr='textured',

                                            vertexPos_input=VertexPosConst,
                                            vertexColor_input=VertexColorConst,
                                            texture_input=VertexTextureConst,
                                            shCoeff_input=SHCConst
                                        ).getRenderBuffer()

    SHC_rnd = tf.Variable(SHCConst+tf.random.uniform([1,1, 27],0, 0.5) )

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    for i in range(100):  
        with tf.GradientTape() as g:
            g.watch(SHC_rnd)
            output = CudaRenderer.CudaRendererGpu(
                                            faces_attr=objreader.facesVertexId,
                                            texCoords_attr=objreader.textureCoordinates,
                                            numberOfVertices_attr=len(objreader.vertexCoordinates),
                                            extrinsics_attr=cameraReader.extrinsics,
                                            intrinsics_attr=cameraReader.intrinsics,
                                            renderResolutionU_attr=1024,
                                            renderResolutionV_attr=1024,
                                            renderMode_attr='textured',
                                            vertexPos_input=VertexPosConst,
                                            vertexColor_input=VertexColorConst,
                                            texture_input=VertexTextureConst,
                                            shCoeff_input=SHC_rnd
                                        ).getRenderBuffer()

            Loss=tf.nn.l2_loss(target-output)
        
        SHC_Grad=g.gradient(Loss,SHC_rnd)
        print(Loss.numpy())

        opt.apply_gradients(zip([SHC_Grad],[SHC_rnd]))

        if(i==0):
            cv.imwrite('test_gradients/targetLighting.png',cv.cvtColor(target[0][0].numpy() * 255.0, cv.COLOR_BGR2RGB))
        if((i+1)%5==0):    
            vertexColorBuffer = output[0][0].numpy() * 255.0
            vertexColorBuffer = cv.cvtColor(vertexColorBuffer, cv.COLOR_BGR2RGB)

            cv.imwrite('test_gradients/SHC {}.png'.format(i),vertexColorBuffer)

########################################################################################################################
# main
########################################################################################################################

freeGPU = CheckGPU.get_free_gpu()

if freeGPU:
    
    test_SHC_gradient()
