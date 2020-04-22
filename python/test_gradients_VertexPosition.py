
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
    numberOfBatches             = 1
    renderResolutionUStart      = 256
    renderResolutionVStart      = 256
    imageFilterSize             = 1
    textureFilterSize           = 1
    shadingMode_attr            = 'shaded'
    albedoMode_attr             = 'textured'
    imageSmoothingSize          = 2
    imageSmoothingStandardDev   = 2.0

    # INPUT
    objreader = OBJReader.OBJReader('data/magdalena.obj')
    cameraReader = CameraReader.CameraReader('data/cameras.calibration', renderResolutionUStart, renderResolutionVStart)

    inputVertexPositions = test_mesh_tensor.getGTMesh()
    # inputVertexPositions = objreader.vertexCoordinates
    inputVertexPositions = np.asarray(inputVertexPositions)
    inputVertexPositions = inputVertexPositions.reshape([1, objreader.numberOfVertices, 3])
    inputVertexPositions = np.tile(inputVertexPositions, (numberOfBatches, 1, 1))

    inputVertexColors = objreader.vertexColors
    inputVertexColors = np.asarray(inputVertexColors)
    inputVertexColors = inputVertexColors.reshape([1, objreader.numberOfVertices, 3])
    inputVertexColors = np.tile(inputVertexColors, (numberOfBatches, 1, 1))
    # inputVertexColors = tf.ones([numberOfBatches, objreader.numberOfVertices, 3])

    inputTexture = objreader.textureMap
    inputTexture = np.asarray(inputTexture)
    inputTexture = inputTexture.reshape([1, objreader.texHeight, objreader.texWidth, 3])
    inputTexture = np.tile(inputTexture, (numberOfBatches, 1, 1, 1))

    inputSHCoeff = test_SH_tensor.getSHCoeff(numberOfBatches, cameraReader.numberOfCameras)

    # objreaderMod = OBJReader.OBJReader('data/coneMod.obj')
    # inputVertexPositionsMod = objreaderMod.vertexCoordinates
    # inputVertexPositionsMod = np.asarray(inputVertexPositionsMod)
    # inputVertexPositionsMod = inputVertexPositionsMod.reshape([1, objreaderMod.numberOfVertices, 3])
    # inputVertexPositionsMod = np.tile(inputVertexPositionsMod, (numberOfBatches, 1, 1))

    VertexPosConst = tf.constant(inputVertexPositions, dtype=tf.float32) * 1.6
    VertexColorConst = tf.constant(inputVertexColors, dtype=tf.float32)
    VertexTextureConst = tf.constant(inputTexture, dtype=tf.float32)
    SHCConst = tf.constant(inputSHCoeff, dtype=tf.float32)

    spatialLoss = LaplacianLoss.LaplacianLoss('data/magdalena.obj', VertexPosConst)

    # generate perturbed vertex positions
    offset = tf.constant([0.0, 50.0, 0.0])
    offset = tf.reshape(offset, [1, 1, 3])
    offset = tf.tile(offset, [numberOfBatches, objreader.numberOfVertices, 1])
    VertexPosition_rnd = tf.Variable((VertexPosConst + tf.random.uniform(tf.shape(VertexPosConst), -50.0, 50.0) + offset), dtype=tf.float32)
    # VertexPosition_rnd = tf.Variable(inputVertexPositionsMod, dtype=tf.float32)

    opt = tf.keras.optimizers.SGD(learning_rate=1.0)

    # sampling levels
    for p in [1,2,4]:

        renderResolutionU = renderResolutionUStart * p
        renderResolutionV = renderResolutionVStart * p
        cameraReader = CameraReader.CameraReader('data/cameras.calibration', renderResolutionU, renderResolutionV)

        rendererTarget = CudaRenderer.CudaRendererGpu(
                                            faces_attr                   = objreader.facesVertexId,
                                            texCoords_attr               = objreader.textureCoordinates,
                                            numberOfVertices_attr        = len(objreader.vertexCoordinates),
                                            extrinsics_attr              = cameraReader.extrinsics ,
                                            intrinsics_attr              = cameraReader.intrinsics,
                                            renderResolutionU_attr       = renderResolutionU ,
                                            renderResolutionV_attr       = renderResolutionV ,
                                            albedoMode_attr              = albedoMode_attr,
                                            shadingMode_attr             = shadingMode_attr,
                                            image_filter_size_attr       = imageFilterSize,
                                            texture_filter_size_attr     = textureFilterSize,

                                            vertexPos_input              = VertexPosConst,
                                            vertexColor_input            = VertexColorConst,
                                            texture_input                = VertexTextureConst,
                                            shCoeff_input                = SHCConst,
                                            targetImage_input            = tf.zeros([numberOfBatches,cameraReader.numberOfCameras,renderResolutionV,renderResolutionU, 3])
                                            )
        targetOriginal = rendererTarget.getRenderBufferTF()

        # Smooth 1
        target = rendererTarget.smoothImage(imageSmoothingSize, 0, imageSmoothingStandardDev)

        for i in range(2500):

            with tf.GradientTape() as tape:

                #render image
                renderer = CudaRenderer.CudaRendererGpu(
                    faces_attr                  = objreader.facesVertexId,
                    texCoords_attr              = objreader.textureCoordinates,
                    numberOfVertices_attr       = len(objreader.vertexCoordinates),
                    extrinsics_attr             = cameraReader.extrinsics,
                    intrinsics_attr             = cameraReader.intrinsics,
                    renderResolutionU_attr      = renderResolutionU,
                    renderResolutionV_attr      = renderResolutionV,
                    albedoMode_attr             = albedoMode_attr,
                    shadingMode_attr            = shadingMode_attr,
                    image_filter_size_attr      = imageFilterSize,
                    texture_filter_size_attr    = textureFilterSize,

                    vertexPos_input             = VertexPosition_rnd,
                    vertexColor_input           = VertexColorConst,
                    texture_input               = VertexTextureConst,
                    shCoeff_input               = SHCConst,
                    targetImage_input           = target
                )
                outputOriginal = renderer.getRenderBufferTF()

                #smooth image
                output = renderer.smoothImage(imageSmoothingSize, 0, imageSmoothingStandardDev)

                #image loss
                difference = (output - target)
                imageLoss = 1.0  * tf.nn.l2_loss((difference) ) / (float(cameraReader.numberOfCameras * 0.25 * renderResolutionV * renderResolutionU))

                #spatial loss
                spatialLossEval = 500.0 * spatialLoss.getLoss(VertexPosition_rnd)

                #combined loss
                loss = imageLoss + spatialLossEval

                #gt loss
                gtLoss = tf.reduce_sum(tf.sqrt( tf.reduce_sum((VertexPosition_rnd - VertexPosConst) * ( VertexPosition_rnd - VertexPosConst),2))).numpy() / float(objreader.numberOfVertices)

            # print loss
            print('Iter '+ str(i) + ' | Loss '+ str(loss.numpy()) + ' | Image '+ str(imageLoss.numpy()) + ' | Spatial '+ str(spatialLossEval.numpy()) + '       |        GTLoss '+ str(gtLoss))

            #output images
            outputCV = cv.cvtColor(outputOriginal[0][1].numpy(), cv.COLOR_RGB2BGR) / 255.0
            targetCV = cv.cvtColor(targetOriginal[0][1].numpy(), cv.COLOR_RGB2BGR) / 255.0
            combined = outputCV
            cv.addWeighted(outputCV, 0.5, targetCV, 0.5, 0.0, combined)
            combined = cv.resize(combined, (1024,1024))
         #   cv.imshow('combined',combined)
            cv.imwrite('test_gradients/image_'+str(i) + '.png',combined * 255)
          #  cv.waitKey(1)

            #apply gradient
            Color_Grad = tape.gradient(loss, VertexPosition_rnd)
            opt.apply_gradients(zip([Color_Grad], [VertexPosition_rnd]))

########################################################################################################################
# main
########################################################################################################################

freeGPU = CheckGPU.get_free_gpu()

if freeGPU:
    test_color_gradient()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    