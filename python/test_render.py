
########################################################################################################################
# Imports
########################################################################################################################

import data.test_mesh_tensor as test_mesh_tensor
import data.test_SH_tensor as test_SH_tensor
import CudaRenderer
import utils.CheckGPU as CheckGPU
import cv2 as cv
import numpy as np
import utils.OBJReader as OBJReader
import utils.CameraReader as CameraReader
import tensorflow as tf

freeGPU = CheckGPU.get_free_gpu()

########################################################################################################################
# CudaRendererGpu class
########################################################################################################################

numberOfBatches = 2
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
# Test render
########################################################################################################################

if freeGPU:

    VertexPosConst = tf.constant(inputVertexPositions, dtype=tf.float32)
    VertexColorConst = tf.constant(inputVertexColors, dtype=tf.float32)
    VertexTextureConst = tf.constant(inputTexture, dtype=tf.float32)
    SHCConst = tf.constant(inputSHCoeff, dtype=tf.float32)

    renderer = CudaRenderer.CudaRendererGpu(
                                            faces_attr                  = objreader.facesVertexId,
                                            texCoords_attr              = objreader.textureCoordinates,
                                            numberOfVertices_attr       = len(objreader.vertexCoordinates),
                                            extrinsics_attr             = cameraReader.extrinsics,
                                            intrinsics_attr             = cameraReader.intrinsics,
                                            renderResolutionU_attr      = renderResolutionU,
                                            renderResolutionV_attr      = renderResolutionV,
                                            renderMode_attr             = 'vertexColor',

                                            vertexPos_input             = VertexPosConst,
                                            vertexColor_input           = VertexColorConst,
                                            texture_input               = VertexTextureConst,
                                            shCoeff_input               = SHCConst,

                                            nodeName                    = 'test')

    # output images
    outputCV = renderer.getRenderBufferOpenCV(1, 0)
    cv.imshow('output', outputCV)
    cv.waitKey(-1)
