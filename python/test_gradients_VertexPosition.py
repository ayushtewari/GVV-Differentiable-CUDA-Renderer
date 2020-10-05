
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from sys import platform
import data.test_mesh_tensor as test_mesh_tensor
import data.test_SH_tensor as test_SH_tensor
import CudaRenderer
import utils.LaplacianLoss as LaplacianLoss
import utils.CheckGPU as CheckGPU
import utils.OBJReader as OBJReader
import utils.CameraReader as CameraReader
import cv2 as cv
import numpy as np
import utils.GaussianSmoothingGpu as GaussianSmoothing

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
# Test color function
########################################################################################################################

def test_color_gradient():

    # HYPERPARAMS
    numberOfBatches             = 3
    renderResolutionUStart      = 512
    renderResolutionVStart      = 512
    imageFilterSize             = 1
    textureFilterSize           = 1
    shadingMode_attr            = 'shaded'
    albedoMode_attr             = 'textured'
    imageSmoothingSize          = 1
    imageSmoothingStandardDev   = 1.0

    # INPUT
    objreader = OBJReader.OBJReader('data/magdalena.obj')
    cameraReader = CameraReader.CameraReader('data/cameras.calibration', renderResolutionUStart, renderResolutionVStart)

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

    VertexPosConst = tf.constant(inputVertexPositions, dtype=tf.float32) * 1.6
    VertexColorConst = tf.constant(inputVertexColors, dtype=tf.float32)
    VertexTextureConst = tf.constant(inputTexture, dtype=tf.float32)
    SHCConst = tf.constant(inputSHCoeff, dtype=tf.float32)

    # generate perturbed vertex positions
    offset = tf.constant([0.0, 50.0, 0.0])
    offset = tf.reshape(offset, [1, 1, 3])
    offset = tf.tile(offset, [numberOfBatches, objreader.numberOfVertices, 1])
    VertexPosition_rnd = tf.Variable((VertexPosConst + tf.random.uniform(tf.shape(VertexPosConst), -0.0, 0.0) + offset), dtype=tf.float32)

    opt = tf.keras.optimizers.Adam(learning_rate=1.0)

    # sampling levels
    for p in [1]:

        renderResolutionU = renderResolutionUStart * p
        renderResolutionV = renderResolutionVStart * p
        cameraReader = CameraReader.CameraReader('data/cameras.calibration', renderResolutionU, renderResolutionV)

        rendererTarget = CudaRenderer.CudaRendererGpu(
                                        faces_attr                   = objreader.facesVertexId,
                                        texCoords_attr               = objreader.textureCoordinates,
                                        numberOfVertices_attr        = len(objreader.vertexCoordinates),
                                        numberOfCameras_attr         = cameraReader.numberOfCameras,
                                        renderResolutionU_attr       = renderResolutionU,
                                        renderResolutionV_attr       = renderResolutionV,
                                        albedoMode_attr              = albedoMode_attr,
                                        shadingMode_attr             = shadingMode_attr,

                                        vertexPos_input              = VertexPosConst,
                                        vertexColor_input            = VertexColorConst,
                                        texture_input                = VertexTextureConst,
                                        shCoeff_input                = SHCConst,
                                        targetImage_input            = tf.zeros( [numberOfBatches, cameraReader.numberOfCameras, renderResolutionV, renderResolutionU, 3]),
                                        extrinsics_input             = [cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
                                        intrinsics_input             = [cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
                                        nodeName                     = 'target'
                                    )
        targetOri2 = rendererTarget.getRenderBufferTF()

        for i in range(100):

            with tf.GradientTape() as tape:
                tape.watch(VertexPosition_rnd)

                #render image
                renderer = CudaRenderer.CudaRendererGpu(
                    faces_attr=objreader.facesVertexId,
                    texCoords_attr=objreader.textureCoordinates,
                    numberOfVertices_attr=len(objreader.vertexCoordinates),
                    numberOfCameras_attr        = cameraReader.numberOfCameras,
                    renderResolutionU_attr=renderResolutionU,
                    renderResolutionV_attr=renderResolutionV,
                    albedoMode_attr=albedoMode_attr,
                    shadingMode_attr = shadingMode_attr,

                    vertexPos_input=VertexPosition_rnd,
                    vertexColor_input=VertexColorConst,
                    texture_input=VertexTextureConst,
                    shCoeff_input=SHCConst,
                    targetImage_input=tf.zeros( [numberOfBatches, cameraReader.numberOfCameras, renderResolutionV, renderResolutionU, 3]),
                    extrinsics_input=[cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
                    intrinsics_input=[cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
                    nodeName='train'
                )
                outputOriginal      = renderer.getRenderBufferTF()
                targetOri           = renderer.getTargetBufferTF()

                # Smooth 1
                target = GaussianSmoothing.smoothImage(targetOri, imageSmoothingSize, 0.0, imageSmoothingStandardDev)
                output = GaussianSmoothing.smoothImage(outputOriginal, imageSmoothingSize, 0.0, imageSmoothingStandardDev)

                target = target * 255.0
                output = output * 255.0

                #image loss
                difference = (output - target)

                foregroundPixels = tf.math.count_nonzero(difference)

                imageLoss = 1.0  * tf.nn.l2_loss((difference) ) / (float(foregroundPixels)* 255.0)

                #spatial loss
                spatialLossEval =  50000.0 * LaplacianLoss.getLoss(VertexPosition_rnd, VertexPosConst, tf.constant(objreader.laplacian),tf.constant(objreader.numberOfEdges), objreader.rowWeight)

                #combined loss
                loss = imageLoss + spatialLossEval

            # print loss
            print('Iter '+ str(i) + ' | Loss '+ str(loss.numpy()) + ' | Image '+ str(imageLoss.numpy()) + ' | Spatial '+ str(spatialLossEval.numpy()) )

            #output images
            outputCV = renderer.getRenderBufferOpenCV(1, 0)
            targetCV = rendererTarget.getRenderBufferOpenCV(1, 0)

            combined = targetCV
            cv.addWeighted(outputCV, 0.8, targetCV, 0.2, 0.0, combined)
            combined = cv.resize(combined,(1024,1024))
            cv.imshow('combined', combined)
            cv.waitKey(1)

            #apply gradient
            Color_Grad = tape.gradient(loss, VertexPosition_rnd)
            opt.apply_gradients(zip([Color_Grad], [VertexPosition_rnd]))

########################################################################################################################
# main
########################################################################################################################

freeGPU = CheckGPU.get_free_gpu()

if freeGPU:
    test_color_gradient()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    