
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
cameraReader = CameraReader.CameraReader('data/cameras.calibration')
testSHCoeff = test_SH_tensor.getSHCoeff(cameraReader.numberOfCameras)
objreader = OBJReader.OBJReader('data/magdalena.obj')

########################################################################################################################
# Test color function
########################################################################################################################

def test_color_gradient():

    VertexPosConst=tf.constant(testMesh3D,dtype=tf.float32)
    VertexColorConst=tf.constant([objreader.vertexColors],dtype=tf.float32)
    VertexTextureConst=tf.constant([objreader.textureMap],dtype=tf.float32)
    SHCConst = tf.constant(testSHCoeff,dtype=tf.float32)

    rendererTarget = CudaRenderer.CudaRendererGpu(
                                        faces_attr                   = objreader.facesVertexId,
                                        texCoords_attr               = objreader.textureCoordinates,
                                        numberOfVertices_attr        = len(objreader.vertexCoordinates),
                                        extrinsics_attr              = cameraReader.extrinsics ,
                                        intrinsics_attr              = cameraReader.intrinsics,
                                        renderResolutionU_attr       = 1024,
                                        renderResolutionV_attr       = 1024,
                                        renderMode_attr              = 'vertexColor',

                                        vertexPos_input              = VertexPosConst,
                                        vertexColor_input            = VertexColorConst,
                                        texture_input                = VertexTextureConst,
                                        shCoeff_input                = SHCConst
                                        )

    target = rendererTarget.getRenderBufferTF()

    VertexColor_rnd = tf.Variable(tf.zeros(VertexColorConst.shape))

    opt = tf.keras.optimizers.SGD(learning_rate=10.0)

    for i in range(3000):

        with tf.GradientTape() as tape:
            tape.watch(VertexColor_rnd)
            renderer = CudaRenderer.CudaRendererGpu(
                faces_attr=objreader.facesVertexId,
                texCoords_attr=objreader.textureCoordinates,
                numberOfVertices_attr=len(objreader.vertexCoordinates),
                extrinsics_attr=cameraReader.extrinsics,
                intrinsics_attr=cameraReader.intrinsics,
                renderResolutionU_attr=1024,
                renderResolutionV_attr=1024,
                renderMode_attr='vertexColor',

                vertexPos_input=VertexPosConst,
                vertexColor_input=VertexColor_rnd,
                texture_input=VertexTextureConst,
                shCoeff_input=SHCConst
            )

            output = renderer.getRenderBufferTF()

            Loss1 = (output-target) * (output-target)
            Loss = tf.reduce_sum(Loss1) / (float(cameraReader.numberOfCameras) * float(objreader.numberOfVertices))

        #apply gradient
        Color_Grad = tape.gradient(Loss,VertexColor_rnd)
        opt.apply_gradients(zip([Color_Grad],[VertexColor_rnd]))

        # print loss
        print(i, Loss.numpy())

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
    test_color_gradient()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    