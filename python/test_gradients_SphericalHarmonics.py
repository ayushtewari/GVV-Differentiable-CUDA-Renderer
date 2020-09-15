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
import numpy as np
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

numberOfBatches = 3
renderResolutionU = 1024
renderResolutionV = 1024

cameraReader = CameraReader.CameraReader('data/cameras.calibration',renderResolutionU,renderResolutionV)
objreader = OBJReader.OBJReader('data/magdalena.obj')

inputVertexPositions = test_mesh_tensor.getGTMesh()
inputVertexPositions = np.asarray(inputVertexPositions)
inputVertexPositions = inputVertexPositions.reshape([1, objreader.numberOfVertices, 3])
inputVertexPositions = np.tile(inputVertexPositions, (numberOfBatches, 1, 1))

inputVertexColors = objreader.vertexColors
inputVertexColors = np.asarray(inputVertexColors)
inputVertexColors = inputVertexColors.reshape([1, objreader.numberOfVertices, 3])
inputVertexColors = np.tile(inputVertexColors, (numberOfBatches, 1, 1))

inputTexture = objreader.textureMap
inputTexture = np.asarray(inputTexture)
inputTexture = inputTexture.reshape([1, objreader.texHeight, objreader.texWidth, 3])
inputTexture = np.tile(inputTexture, (numberOfBatches, 1, 1, 1))

inputSHCoeff = test_SH_tensor.getSHCoeff(numberOfBatches, cameraReader.numberOfCameras)

########################################################################################################################
# test spherical harmonics lighting
########################################################################################################################

def test_SHC_gradient():

    VertexPosConst          = tf.constant(inputVertexPositions,     dtype=tf.float32)
    VertexColorConst        = tf.constant(inputVertexColors,        dtype=tf.float32)
    VertexTextureConst      = tf.constant(inputTexture,             dtype=tf.float32)
    SHCConst                = tf.constant(inputSHCoeff,             dtype=tf.float32)

    rendererTarget = CudaRenderer.CudaRendererGpu(
                                        faces_attr                   = objreader.facesVertexId,
                                        texCoords_attr               = objreader.textureCoordinates,
                                        numberOfVertices_attr        = len(objreader.vertexCoordinates),
                                        numberOfCameras_attr         = cameraReader.numberOfCameras,
                                        renderResolutionU_attr       = renderResolutionU,
                                        renderResolutionV_attr       = renderResolutionV,
                                        albedoMode_attr              = 'vertexColor',
                                        shadingMode_attr             = 'shaded',

                                        vertexPos_input              = VertexPosConst,
                                        vertexColor_input            = VertexColorConst,
                                        texture_input                = VertexTextureConst,
                                        shCoeff_input                = SHCConst,
                                        targetImage_input            = tf.zeros( [numberOfBatches, cameraReader.numberOfCameras, renderResolutionV, renderResolutionU, 3]),
                                        extrinsics_input             = [cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
                                        intrinsics_input             = [cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
                                        nodeName                     = 'target'
                                    )
    target = rendererTarget.getRenderBufferTF()

    SHC_rnd = tf.Variable(SHCConst+tf.random.uniform([1,1, 27],0, 0.5) )

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    for i in range(10):
        with tf.GradientTape() as g:
            g.watch(SHC_rnd)
            renderer = CudaRenderer.CudaRendererGpu(
                faces_attr=objreader.facesVertexId,
                texCoords_attr=objreader.textureCoordinates,
                numberOfVertices_attr=len(objreader.vertexCoordinates),
                numberOfCameras_attr        = cameraReader.numberOfCameras,
                renderResolutionU_attr=renderResolutionU,
                renderResolutionV_attr=renderResolutionV,
                albedoMode_attr='vertexColor',
                shadingMode_attr='shaded',

                vertexPos_input=VertexPosConst,
                vertexColor_input=VertexColorConst,
                texture_input=VertexTextureConst,
                shCoeff_input=SHC_rnd,
                targetImage_input=tf.zeros( [numberOfBatches, cameraReader.numberOfCameras, renderResolutionV, renderResolutionU, 3]),
                extrinsics_input=[cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
                intrinsics_input=[cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
                nodeName='train'
            )

            output = renderer.getRenderBufferTF()

            Loss=tf.nn.l2_loss(target-output)

        # apply gradient
        SHC_Grad=g.gradient(Loss,SHC_rnd)
        print(SHC_Grad)
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
