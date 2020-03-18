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

    rendererTarget = CudaRenderer.CudaRendererGpu(
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
                                        )

    target = rendererTarget.getRenderBufferTF()

    SHC_rnd = tf.Variable(SHCConst+tf.random.uniform([1,1, 27],0, 0.5) )

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    for i in range(2000):
        with tf.GradientTape() as g:
            g.watch(SHC_rnd)
            renderer = CudaRenderer.CudaRendererGpu(
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
                                        )
            output = renderer.getRenderBufferTF()

            Loss=tf.nn.l2_loss(target-output)

        # apply gradient
        SHC_Grad=g.gradient(Loss,SHC_rnd)
        opt.apply_gradients(zip([SHC_Grad], [SHC_rnd]))

        # print loss
        print(Loss.numpy())

        # output images
        outputCV = renderer.getRenderBufferOpenCV(0, 0)
        targetCV = rendererTarget.getRenderBufferOpenCV(0, 0)

        combined = targetCV
        cv.addWeighted(outputCV, 0.8, targetCV, 0.2, 0.0, combined)
        cv.imshow('combined', combined)
        cv.waitKey(1)

########################################################################################################################
# main
########################################################################################################################

freeGPU = CheckGPU.get_free_gpu()

if freeGPU:
    
    test_SHC_gradient()
