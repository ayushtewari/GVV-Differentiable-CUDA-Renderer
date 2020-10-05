
import tensorflow as tf

########################################################################################################################
# Isometry Loss
########################################################################################################################

def getLoss(inputMeshTensor, restTensor, laplacian, numberOfEdges, rowWeight):

    batchSize = tf.shape(inputMeshTensor)[0]
    numberOfVertices = tf.shape(inputMeshTensor)[1]

    v_r = (inputMeshTensor/1000.0) - (restTensor/1000.0)

    innerSumX = tf.matmul( laplacian,  tf.reshape(v_r[:, :, 0], [batchSize,numberOfVertices, 1]))
    innerSumX = innerSumX * innerSumX

    innerSumY = tf.matmul(laplacian,  tf.reshape(v_r[:, :, 1], [batchSize,numberOfVertices, 1]))
    innerSumY = innerSumY * innerSumY

    innerSumZ = tf.matmul(laplacian,  tf.reshape(v_r[:, :, 2], [batchSize,numberOfVertices, 1]))
    innerSumZ = innerSumZ * innerSumZ

    innerSum = innerSumX + innerSumY + innerSumZ

    innerSum = tf.reshape(innerSum,[batchSize,numberOfVertices])

    loss = tf.reduce_sum(innerSum * rowWeight)

    loss = loss / tf.cast(batchSize * numberOfEdges,tf.float32)

    return loss

########################################################################################################################
