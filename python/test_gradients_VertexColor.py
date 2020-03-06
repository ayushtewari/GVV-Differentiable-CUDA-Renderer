
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

    target = CudaRenderer.CudaRendererGpu(
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
                                        ).getRenderBuffer()

    VertexColor_rnd = tf.Variable(tf.zeros(VertexColorConst.shape))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        1e+1,
        decay_steps=6000,
        decay_rate=0.1,
        staircase=True)

    opt = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)

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

            output = renderer.getRenderBuffer()


            Loss = tf.reduce_sum(tf.keras.losses.mean_absolute_error(target,output)) / (float(cameraReader.numberOfCameras) * float(objreader.numberOfVertices))

        #print loss
        print(i, Loss.numpy())

        #apply gradient
        Color_Grad= tape.gradient(Loss,VertexColor_rnd)
        opt.apply_gradients(zip([Color_Grad],[VertexColor_rnd]))
        VertexColor_rnd = tf.Variable(tf.clip_by_value(VertexColor_rnd, 0, 1))

        #output images
        cv.imwrite('test_gradients/target.png',cv.cvtColor(target[0][0].numpy() * 255.0, cv.COLOR_BGR2RGB))

        vertexColorBuffer = output[0][0].numpy() * 255.0
        vertexColorBuffer = cv.cvtColor(vertexColorBuffer, cv.COLOR_BGR2RGB)
        cv.imwrite('test_gradients/Color {}.png'.format(i),vertexColorBuffer)

########################################################################################################################
# main
########################################################################################################################

freeGPU = CheckGPU.get_free_gpu()

if freeGPU:
    test_color_gradient()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    