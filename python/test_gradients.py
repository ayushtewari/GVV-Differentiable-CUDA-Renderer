########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
from sys import platform
import data.test_mesh_tensor as test_mesh_tensor
import data.test_SH_tensor as test_SH_tensor
import numpy as np

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
testSHCoeff = test_SH_tensor.getSHCoeff()

objreader = OBJReader.OBJReader('data/magdalena.obj')
cameraReader = CameraReader.CameraReader('data/test.calibration')

@ops.RegisterGradient("CudaRendererGpu")
def cuda_renderer_gpu_grad(op, gradBarycentric, gradFace, gradDepth, gradRender, gradVertexColor,gradBoundry,gradVisibility,gradNorm):
    
    #return customOperators.cuda_renderer_grad_gpu(grad,*op.inputs)
    TextureZeroGrad = tf.zeros(tf.shape(op.inputs[2]), tf.float32)
    gradients=customOperators.cuda_renderer_grad_gpu( vertex_color_buffer_grad=gradVertexColor,
                                                                        faces                   = objreader.facesVertexId,
                                                                        texture_coordinates     = objreader.textureCoordinates,
                                                                        number_of_vertices      = len(objreader.vertexCoordinates),
                                                                        extrinsics              = cameraReader.extrinsics ,
                                                                        intrinsics              = cameraReader.intrinsics,
                                                                        render_resolution_u     = 1024,
                                                                        render_resolution_v     = 1024,

                                                                        vertex_pos              = op.inputs[0],
                                                                        vertex_color            = op.inputs[1],
                                                                        texture                 = op.inputs[2],
                                                                        sh_coeff                = op.inputs[3],
                                                                        
                                                                        vertex_normal           = op.outputs[7],
                                                                        barycentric_buffer      = op.outputs[0],
                                                                        face_buffer             = op.outputs[1],
                                                                        vertex_color_buffer     = op.outputs[4])
    return[gradients[0],gradients[1],TextureZeroGrad,gradients[2]]
    

def test_color_gradient():
    batch_size=1
    VertexPosConst=tf.constant(testMesh3D,dtype=tf.float32)
    VertexColorConst=tf.constant([objreader.vertexColors],dtype=tf.float32)
    VertexTextureConst=tf.constant([objreader.textureMap],dtype=tf.float32)
    #vertexColor_rnd=
    SHCConst=tf.constant(testSHCoeff,dtype=tf.float32)
    target = customOperators.cuda_renderer_gpu( 
                                                                            faces                   = objreader.facesVertexId,
                                                                            texture_coordinates     = objreader.textureCoordinates,
                                                                            number_of_vertices      = len(objreader.vertexCoordinates),
                                                                            extrinsics              = cameraReader.extrinsics ,
                                                                            intrinsics              = cameraReader.intrinsics,
                                                                            render_resolution_u     = 1024,
                                                                            render_resolution_v     = 1024,
    
                                                                            vertex_pos              = VertexPosConst,
                                                                            vertex_color            = VertexColorConst,
                                                                            texture                 = VertexTextureConst,
                                                                            sh_coeff                = SHCConst
                                                                            )[4][0][0]
    
    VertexColor_rnd=tf.Variable(tf.zeros(VertexColorConst.shape))
    #VertexColor_rnd=tf.Variable(VertexColorConst+)
####Optimizition    
#    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    1e-2,
#    decay_steps=1000,
#    decay_rate=0.8,
#    staircase=True)


    opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
    for i in range(200):  
        with tf.GradientTape() as g:
            g.watch(VertexColor_rnd)
            output = customOperators.cuda_renderer_gpu( 
                                                                            faces                   = objreader.facesVertexId,
                                                                            texture_coordinates     = objreader.textureCoordinates,
                                                                            number_of_vertices      = len(objreader.vertexCoordinates),
                                                                            extrinsics              = cameraReader.extrinsics ,
                                                                            intrinsics              = cameraReader.intrinsics,
                                                                            render_resolution_u     = 1024,
                                                                            render_resolution_v     = 1024,
        
                                                                            vertex_pos              = VertexPosConst,
                                                                            vertex_color            = VertexColor_rnd,
                                                                            texture                 = VertexTextureConst,
                                                                            sh_coeff                = SHCConst
                                                                            )[4][0][0]
        
            #regularizer=tf.keras.regularizers.l2(0.05)
            #regularizer(VertexColor_rnd)
            Loss=tf.nn.l2_loss((target-output) )
        
        Color_Grad=g.gradient(Loss,VertexColor_rnd)
        print(Loss.numpy())
        #print(grad_SHC)
        #print(grad_SHC.shape,SHC_rnd.shape)
        opt.apply_gradients(zip([Color_Grad],[VertexColor_rnd]))
        #VertexColor_rnd=tf.Variable(tf.clip_by_value(VertexColorConst,0.0,1.0))
        
        if(i==0):
            cv.imwrite('test_gradients/target.png',cv.cvtColor(target.numpy() * 255.0, cv.COLOR_BGR2RGB))
        if((i+1)%5==0):    
            vertexColorBuffer = output.numpy() * 255.0
            vertexColorBuffer = cv.cvtColor(vertexColorBuffer, cv.COLOR_BGR2RGB)
            #opt.minimize(Loss, var_list=[SHC_rnd])
            cv.imwrite('test_gradients/Color {}.png'.format(i),vertexColorBuffer)

def test_SHC_gradient():
    batch_size=1
    VertexPosConst=tf.constant(testMesh3D,dtype=tf.float32)
    VertexColorConst=tf.constant([objreader.vertexColors],dtype=tf.float32)
    VertexTextureConst=tf.constant([objreader.textureMap],dtype=tf.float32)
    SHCConst=tf.constant(testSHCoeff,dtype=tf.float32)
    target = customOperators.cuda_renderer_gpu( 
                                                                            faces                   = objreader.facesVertexId,
                                                                            texture_coordinates     = objreader.textureCoordinates,
                                                                            number_of_vertices      = len(objreader.vertexCoordinates),
                                                                            extrinsics              = cameraReader.extrinsics ,
                                                                            intrinsics              = cameraReader.intrinsics,
                                                                            render_resolution_u     = 1024,
                                                                            render_resolution_v     = 1024,
    
                                                                            vertex_pos              = VertexPosConst,
                                                                            vertex_color            = VertexColorConst,
                                                                            texture                 = VertexTextureConst,
                                                                            sh_coeff                = SHCConst
                                                                            )[4][0][0]
    SHC_rnd = tf.Variable(SHCConst+tf.random.uniform([batch_size,1, 27],0, 0.5) )

####Optimizition    
#    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    1e-2,
#    decay_steps=1000,
#    decay_rate=0.8,
#    staircase=True)


    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    for i in range(100):  
        with tf.GradientTape() as g:
            g.watch(SHC_rnd)
            output = customOperators.cuda_renderer_gpu( 
                                                                            faces                   = objreader.facesVertexId,
                                                                            texture_coordinates     = objreader.textureCoordinates,
                                                                            number_of_vertices      = len(objreader.vertexCoordinates),
                                                                            extrinsics              = cameraReader.extrinsics ,
                                                                            intrinsics              = cameraReader.intrinsics,
                                                                            render_resolution_u     = 1024,
                                                                            render_resolution_v     = 1024,
        
                                                                            vertex_pos              = VertexPosConst,
                                                                            vertex_color            = VertexColorConst,
                                                                            texture                 = VertexTextureConst,
                                                                            sh_coeff                = SHC_rnd
                                                                            )[4][0][0]
        
            #regularizer=tf.keras.regularizers.l1(0.05)
            #regularizer(SHC_rnd)
            Loss=tf.nn.l2_loss(target-output)
        
        SHC_Grad=g.gradient(Loss,SHC_rnd)
        print(Loss.numpy())
        #print(grad_SHC)
        #print(grad_SHC.shape,SHC_rnd.shape)
        opt.apply_gradients(zip([SHC_Grad],[SHC_rnd]))
        
        
        if(i==0):
            cv.imwrite('test_gradients/target.png',cv.cvtColor(target.numpy() * 255.0, cv.COLOR_BGR2RGB))
        if((i+1)%5==0):    
            vertexColorBuffer = output.numpy() * 255.0
            vertexColorBuffer = cv.cvtColor(vertexColorBuffer, cv.COLOR_BGR2RGB)
            #opt.minimize(Loss, var_list=[SHC_rnd])
            cv.imwrite('test_gradients/SHC {}.png'.format(i),vertexColorBuffer)
freeGPU = CheckGPU.get_free_gpu()

if freeGPU:
    
    test_color_gradient()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    