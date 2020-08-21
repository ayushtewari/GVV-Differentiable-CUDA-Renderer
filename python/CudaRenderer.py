
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
from sys import platform
import cv2 as cv
import numpy as np
import tensorflow_probability as tfp

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

    ########################################################################################################################

    def __init__(self,
                 faces_attr                 = [],
                 texCoords_attr             = [],
                 numberOfVertices_attr      = -1,
                 numberOfCameras_attr       = -1,
                 renderResolutionU_attr     = -1,
                 renderResolutionV_attr     = -1,
                 albedoMode_attr            = 'textured',
                 shadingMode_attr           = 'shaded',
                 image_filter_size_attr     = 1,
                 texture_filter_size_attr   = 1,
                 compute_normal_map_attr    = False,

                 vertexPos_input            = None,
                 vertexColor_input          = None,
                 texture_input              = None,
                 shCoeff_input              = None,
                 targetImage_input          = None,
                 extrinsics_input           = [],
                 intrinsics_input           = [],

                 nodeName                   = 'CudaRenderer'):

        self.faces_attr                 = faces_attr
        self.texCoords_attr             = texCoords_attr
        self.numberOfVertices_attr      = numberOfVertices_attr
        self.numberOfCameras_attr       = numberOfCameras_attr
        self.renderResolutionU_attr     = renderResolutionU_attr
        self.renderResolutionV_attr     = renderResolutionV_attr
        self.albedoMode_attr            = albedoMode_attr
        self.shadingMode_attr           = shadingMode_attr
        self.image_filter_size_attr     = image_filter_size_attr
        self.texture_filter_size_attr   = texture_filter_size_attr
        self.compute_normal_map_attr    = compute_normal_map_attr

        self.vertexPos_input            = vertexPos_input
        self.vertexColor_input          = vertexColor_input
        self.texture_input              = texture_input
        self.shCoeff_input              = shCoeff_input
        self.targetImage_input          = targetImage_input
        self.extrinsics_input           = extrinsics_input
        self.intrinsics_input           = intrinsics_input

        self.nodeName                   = nodeName

        self.cudaRendererOperator = customOperators.cuda_renderer_gpu(  faces                   = self.faces_attr,
                                                                        texture_coordinates     = self.texCoords_attr,
                                                                        number_of_vertices      = self.numberOfVertices_attr,
                                                                        number_of_cameras       = self.numberOfCameras_attr,
                                                                        render_resolution_u     = self.renderResolutionU_attr,
                                                                        render_resolution_v     = self.renderResolutionV_attr,
                                                                        albedo_mode             = self.albedoMode_attr,
                                                                        shading_mode            = self.shadingMode_attr,
                                                                        image_filter_size       = self.image_filter_size_attr,
                                                                        texture_filter_size     = self.texture_filter_size_attr,
                                                                        compute_normal_map      = self.compute_normal_map_attr,

                                                                        vertex_pos              = self.vertexPos_input,
                                                                        vertex_color            = self.vertexColor_input,
                                                                        texture                 = self.texture_input,
                                                                        sh_coeff                = self.shCoeff_input,
                                                                        target_image            = self.targetImage_input,
                                                                        extrinsics              = self.extrinsics_input,
                                                                        intrinsics              = self.intrinsics_input,

                                                                        name                    = self.nodeName)

    ########################################################################################################################

    def getBaryCentricBufferTF(self):
        return self.cudaRendererOperator[0]

    ########################################################################################################################

    def getFaceBufferTF(self):
        return self.cudaRendererOperator[1]

    ########################################################################################################################

    def getRenderBufferTF(self):
        return self.cudaRendererOperator[2]

    ########################################################################################################################

    def getTargetBufferTF(self):
        return self.cudaRendererOperator[4]

    ########################################################################################################################

    def getNormalMap(self):
        if self.compute_normal_map_attr:
            normalMap = self.cudaRendererOperator[5]
            normalMap = tf.reshape(normalMap, tf.shape(self.texture_input))
            return normalMap
        else:
            tf.print('Requesting normal map but computation was not enabled!')
            return None

    ########################################################################################################################

    def getModelMaskTF(self):
        shape = tf.shape(self.cudaRendererOperator[1])
        mask = tf.greater_equal(self.cudaRendererOperator[1], 0)
        mask = tf.reshape(mask, [shape[0], shape[1] , shape[2], shape[3], 1])
        mask = tf.tile(mask, [1, 1, 1, 1, 3])
        mask = tf.cast(mask, tf.float32)
        return mask

    ########################################################################################################################

    def getBaryCentricBufferOpenCV(self, batchId, camId):
        return cv.cvtColor(self.cudaRendererOperator[0][batchId][camId].numpy(), cv.COLOR_RGB2BGR)

    ########################################################################################################################

    def getFaceBufferOpenCV(self, batchId, camId):
        faceImg = self.cudaRendererOperator[1][batchId][camId].numpy().astype(np.float32)    #convert to float
        faceImg = faceImg[:,:,0]   + 1.0                                                     #only select the face channel
        return cv.cvtColor(faceImg, cv.COLOR_GRAY2RGB)                                       #convert grey to rgb for visualization

    ########################################################################################################################

    def getRenderBufferOpenCV(self, batchId, camId):
        return  cv.cvtColor(self.cudaRendererOperator[2][batchId][camId].numpy(), cv.COLOR_RGB2BGR)

    ########################################################################################################################

    def getNormalMapOpenCV(self, batchId):
        return  cv.cvtColor(self.cudaRendererOperator[5][batchId].numpy(), cv.COLOR_RGB2BGR)

    ########################################################################################################################

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("CudaRendererGpu")
def cuda_renderer_gpu_grad(op, gradBarycentric, gradFace, gradRender, gradNorm, gradTarget, gradNormalMap):

    albedoMode = op.get_attr('albedo_mode').decode("utf-8")

    if(albedoMode == 'vertexColor' or albedoMode == 'textured' or albedoMode == 'foregroundMask'):
        gradients = customOperators.cuda_renderer_grad_gpu(
            # grads
            render_buffer_grad          = gradRender,
            target_buffer_grad          = gradTarget,

            # inputs
            vertex_pos                  = op.inputs[0],
            vertex_color                = op.inputs[1],
            texture                     = op.inputs[2],
            sh_coeff                    = op.inputs[3],
            target_image                = op.inputs[4],
            extrinsics                  = op.inputs[5],
            intrinsics                  = op.inputs[6],

            barycentric_buffer          = op.outputs[0],
            face_buffer                 = op.outputs[1],
            vertex_normal               = op.outputs[3],


            # attr
            faces                       = op.get_attr('faces'),
            texture_coordinates         = op.get_attr('texture_coordinates'),
            number_of_vertices          = op.get_attr('number_of_vertices'),
            number_of_cameras           = op.get_attr('number_of_cameras'),
            render_resolution_u         = op.get_attr('render_resolution_u'),
            render_resolution_v         = op.get_attr('render_resolution_v'),
            albedo_mode                 = op.get_attr('albedo_mode'),
            shading_mode                = op.get_attr('shading_mode'),
            image_filter_size           = op.get_attr('image_filter_size'),
            texture_filter_size         = op.get_attr('texture_filter_size')
        )
    elif (albedoMode == 'normal' or albedoMode == 'lighting'):
        gradients = [
            tf.zeros(tf.shape(op.inputs[0])),
            tf.zeros(tf.shape(op.inputs[1])),
            tf.zeros(tf.shape(op.inputs[2])),
            tf.zeros(tf.shape(op.inputs[3])),
        ]

    return gradients[0], gradients[1], gradients[2], gradients[3],  tf.zeros(tf.shape(op.inputs[4])), tf.zeros(tf.shape(op.inputs[5])), tf.zeros(tf.shape(op.inputs[6]))

########################################################################################################################
#
########################################################################################################################