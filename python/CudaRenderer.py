
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops

########################################################################################################################
# Load custom operators
########################################################################################################################

customOperators = tf.load_op_library(r"\\winfs-inf.mpi-inf.mpg.de\HPS\RTMPC\work\CudaRenderer\cpp\binaries\Win64\Release\CustomTensorFlowOperators.dll")

########################################################################################################################
# CudaRendererGpu class
########################################################################################################################

class CudaRendererGpu:

    def __init__(self,
                 cameraFilePath = '',
                 meshFilePath = '',
                 renderResolutionU = 512,
                 renderResolutionV = 512,
                 pointsGlobalSpace = None,
                 nodeName=''):

        self.cameraFilePath = cameraFilePath
        self.meshFilePath = meshFilePath

        self.renderResolutionU = renderResolutionU
        self.renderResolutionV = renderResolutionV

 
        self.pointsGlobalSpace = pointsGlobalSpace
        self.nodeName = nodeName

        self.cudaRendererOperator = None

        if(cameraFilePath != '' and meshFilePath != '' and pointsGlobalSpace is not None and nodeName != ''):

            self.cudaRendererOperator = customOperators.cuda_renderer_gpu(  pointsGlobalSpace,
                                                                            render_resolution_u =  self.renderResolutionU,
                                                                            render_resolution_v = self.renderResolutionV,
                                                                            camera_file_path_boundary_check = cameraFilePath,
                                                                            mesh_file_path_boundary_check = meshFilePath,
                                                                            name=nodeName)

        else:

            raise ValueError('Invalid argument during the construction of the projected mesh boundary operator!')


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


