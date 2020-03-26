
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
from sys import platform
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
# CudaRendererGpu class
########################################################################################################################

class CudaRendererGpu:

    def __init__(self,
                 faces_attr,
                 texCoords_attr,
                 numberOfVertices_attr,
                 extrinsics_attr,
                 intrinsics_attr,
                 renderResolutionU_attr,
                 renderResolutionV_attr,
                 renderMode_attr,

                 vertexPos_input ,
                 vertexColor_input,
                 texture_input,
                 shCoeff_input,

                 nodeName=''):

        self.faces_attr                 = faces_attr
        self.texCoords_attr             = texCoords_attr
        self.numberOfVertices_attr      = numberOfVertices_attr
        self.extrinsics_attr            = extrinsics_attr
        self.intrinsics_attr            = intrinsics_attr
        self.renderResolutionU_attr     = renderResolutionU_attr
        self.renderResolutionV_attr     = renderResolutionV_attr
        self.renderMode_attr            = renderMode_attr

        self.vertexPos_input            = vertexPos_input
        self.vertexColor_input          = vertexColor_input
        self.texture_input              = texture_input
        self.shCoeff_input              = shCoeff_input

        self.nodeName                   = nodeName

        self.cudaRendererOperator = customOperators.cuda_renderer_gpu(  faces                   = self.faces_attr,
                                                                        texture_coordinates     = self.texCoords_attr,
                                                                        number_of_vertices      = self.numberOfVertices_attr,
                                                                        extrinsics              = self.extrinsics_attr ,
                                                                        intrinsics              = self.intrinsics_attr,
                                                                        render_resolution_u     = self.renderResolutionU_attr,
                                                                        render_resolution_v     = self.renderResolutionV_attr,
                                                                        render_mode             = self.renderMode_attr,

                                                                        vertex_pos              = self.vertexPos_input,
                                                                        vertex_color            = self.vertexColor_input,
                                                                        texture                 = self.texture_input,
                                                                        sh_coeff                = self.shCoeff_input,

                                                                        name                    = self.nodeName)

    def getBaryCentricBufferTF(self):
        return self.cudaRendererOperator[0]

    def getFaceBufferTF(self):
        return self.cudaRendererOperator[1]

    def getRenderBufferTF(self):
        return self.cudaRendererOperator[2]

    def getModelMaskTF(self):
        shape = tf.shape(self.cudaRendererOperator[1])
        mask = tf.greater_equal(self.cudaRendererOperator[1], 0)
        mask = tf.reshape(mask, [shape[0], shape[1] , shape[2], shape[3], 1])
        mask = tf.tile(mask, [1, 1, 1, 1, 3])
        mask = tf.cast(mask, tf.float32)
        return mask

    def getBaryCentricBufferOpenCV(self, batchId, camId):
        return cv.cvtColor(self.cudaRendererOperator[0][batchId][camId].numpy(), cv.COLOR_RGB2BGR)

    def getFaceBufferOpenCV(self, batchId, camId):
        faceImg = self.cudaRendererOperator[1][batchId][camId].numpy().astype(np.float32)    #convert to float
        faceImg = faceImg[:,:,0]   + 1.0                                                     #only select the face channel
        return cv.cvtColor(faceImg, cv.COLOR_GRAY2RGB)                                       #convert grey to rgb for visualization

    def getRenderBufferOpenCV(self, batchId, camId):
        return  cv.cvtColor(self.cudaRendererOperator[2][batchId][camId].numpy(), cv.COLOR_RGB2BGR)

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("CudaRendererGpu")
def cuda_renderer_gpu_grad(op, gradBarycentric, gradFace, gradRender, gradNorm):

    renderMode = op.get_attr('render_mode').decode("utf-8")

    if(renderMode == 'vertexColor' or renderMode == 'textured' ):
        gradients = customOperators.cuda_renderer_grad_gpu(
            # grads
            render_buffer_grad          = gradRender,

            # inputs
            vertex_pos                  = op.inputs[0],
            vertex_color                = op.inputs[1],
            texture                     = op.inputs[2],
            sh_coeff                    = op.inputs[3],

            barycentric_buffer          = op.outputs[0],
            face_buffer                 = op.outputs[1],
            vertex_normal               = op.outputs[3],

            # attr
            faces                       = op.get_attr('faces'),
            texture_coordinates         = op.get_attr('texture_coordinates'),
            number_of_vertices          = op.get_attr('number_of_vertices'),
            extrinsics                  = op.get_attr('extrinsics'),
            intrinsics                  = op.get_attr('intrinsics'),
            render_resolution_u         = op.get_attr('render_resolution_u'),
            render_resolution_v         = op.get_attr('render_resolution_v'),
            render_mode                 = op.get_attr('render_mode'),
        )
    elif (renderMode == 'normal'):
        gradients = [
            tf.zeros(tf.shape(op.inputs[0])),
            tf.zeros(tf.shape(op.inputs[1])),
            tf.zeros(tf.shape(op.inputs[2])),
            tf.zeros(tf.shape(op.inputs[3])),
        ]

    return gradients[0], gradients[1], gradients[2], gradients[3]

########################################################################################################################
#
########################################################################################################################
