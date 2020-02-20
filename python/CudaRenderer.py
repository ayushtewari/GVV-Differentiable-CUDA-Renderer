
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
from sys import platform

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

                                                                        vertex_pos              = self.vertexPos_input,
                                                                        vertex_color            = self.vertexColor_input,
                                                                        texture                 = self.texture_input,
                                                                        sh_coeff                = self.shCoeff_input,

                                                                        name                    = self.nodeName)


    def getBaryCentricBuffer(self):
        return self.cudaRendererOperator[0]
    def getFaceBuffer(self):
        return self.cudaRendererOperator[1]
    def getDepthBuffer(self):
        return self.cudaRendererOperator[2]
    def getRenderBuffer(self):
        return self.cudaRendererOperator[3]
    def getVertexColorBuffer(self):
        return self.cudaRendererOperator[4]
    def getVertexNormal(self):
        return self.cudaRendererOperator[7]

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("CudaRendererGpu")
def cuda_renderer_gpu_grad(op, gradBarycentric, gradFace, gradDepth, gradRender, gradVertexColor):

    # determine the zero gradient stuff
    pointsGlobalSpaceZeroGrad = tf.zeros(tf.shape(op.inputs[1]), tf.float32)
    
    return pointsGlobalSpaceZeroGrad

########################################################################################################################
#
########################################################################################################################


