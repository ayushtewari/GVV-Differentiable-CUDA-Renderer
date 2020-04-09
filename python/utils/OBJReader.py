
import cv2
import numpy as np

########################################################################################################################
# OBJ Reader
########################################################################################################################

class OBJReader:

    ########################################################################################################################

    def __init__(self, filename):

        self.filename = filename
        self.folderPath = self.filename[0:self.filename.rfind('/') + 1]

        self.readObjFile()

        self.numberOfVertices = len(self.vertexColors)

        self.computePerFaceTextureCoordinated()

        self.loadSegmentationWeights()

        self.computeAdjacency()

        self.loadMtlTexture(self.mtlFilePathFull, self.mtlFilePath)

    ########################################################################################################################

    def readObjFile(self):

        file = open(self.filename)

        # read faces
        self.facesVertexId = []
        self.facesTextureId = []
        self.vertexColors = []
        self.vertexCoordinates = []
        self.pertVertexTextureCoordinate = []

        for line in file:
            splitted = line.split()
            if len(splitted) > 0:

                # is face
                if splitted[0] == 'f':
                    v0 = splitted[1].split('/')[0]
                    v1 = splitted[2].split('/')[0]
                    v2 = splitted[3].split('/')[0]

                    self.facesVertexId.append(int(v0) - 1)
                    self.facesVertexId.append(int(v1) - 1)
                    self.facesVertexId.append(int(v2) - 1)

                    t0 = splitted[1].split('/')[1]
                    t1 = splitted[2].split('/')[1]
                    t2 = splitted[3].split('/')[1]

                    self.facesTextureId.append(int(t0) - 1)
                    self.facesTextureId.append(int(t1) - 1)
                    self.facesTextureId.append(int(t2) - 1)

                # is vertex
                if splitted[0] == 'v':
                    self.vertexCoordinates.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
                    self.vertexColors.append([float(splitted[4]), float(splitted[5]), float(splitted[6])])

                # is texture coordinate
                if splitted[0] == 'vt':
                    self.pertVertexTextureCoordinate.append([float(splitted[1]), float(splitted[2])])

                # is mtllib
                if splitted[0] == 'mtllib':
                    self.mtlFilePath = self.filename[0:self.filename.rfind('/') + 1]
                    mtlName = splitted[1][2:]
                    self.mtlFilePathFull = self.mtlFilePath + mtlName

        file.close()

    ########################################################################################################################

    def computePerFaceTextureCoordinated(self):
        # per face texture coordinates
        self.textureCoordinates = []
        for t in range(0, len(self.facesTextureId)):
            texCoord = self.pertVertexTextureCoordinate[self.facesTextureId[t]]
            self.textureCoordinates.append(texCoord[0])
            self.textureCoordinates.append(texCoord[1])

    ########################################################################################################################

    def computeAdjacency(self):

        # adjacency matrix
        self.adjacency = np.zeros((self.numberOfVertices, self.numberOfVertices))

        for f in range(0,int(len(self.facesVertexId)/3)):
            v0 = self.facesVertexId[f * 3 + 0]
            v1 = self.facesVertexId[f * 3 + 1]
            v2 = self.facesVertexId[f * 3 + 2]

            self.adjacency[v0, v1] = 1
            self.adjacency[v0, v2] = 1
            self.adjacency[v1, v0] = 1
            self.adjacency[v1, v2] = 1
            self.adjacency[v2, v0] = 1
            self.adjacency[v2, v1] = 1

        # number of edges
        self.numberOfEdges = 0
        for i in range(0, self.numberOfVertices):
            for j in range(0, self.numberOfVertices):
                if(self.adjacency[i,j] > 0.0):
                    self.numberOfEdges = self.numberOfEdges + 1

        #weights matrix
        if len(self.vertexLabels) == self.numberOfVertices:
            self.adjacencyWeights = np.zeros((self.numberOfVertices, self.numberOfVertices))
            for f in range(0, int(len(self.facesVertexId) / 3)):
                v0 = self.facesVertexId[f * 3 + 0]
                v1 = self.facesVertexId[f * 3 + 1]
                v2 = self.facesVertexId[f * 3 + 2]

                self.adjacencyWeights[v0, v1] = (self.vertexWeights[v0] + self.vertexWeights[v1]) / 2.0
                self.adjacencyWeights[v0, v2] = (self.vertexWeights[v0] + self.vertexWeights[v2]) / 2.0
                self.adjacencyWeights[v1, v0] = (self.vertexWeights[v1] + self.vertexWeights[v0]) / 2.0
                self.adjacencyWeights[v1, v2] = (self.vertexWeights[v1] + self.vertexWeights[v2]) / 2.0
                self.adjacencyWeights[v2, v0] = (self.vertexWeights[v2] + self.vertexWeights[v0]) / 2.0
                self.adjacencyWeights[v2, v1] = (self.vertexWeights[v2] + self.vertexWeights[v1]) / 2.0

    ########################################################################################################################

    def loadMtlTexture(self,mtlFileName,shortPath):
        mtlFile = open(mtlFileName)

        for line in mtlFile:
            splitted = line.split()
            if len(splitted) > 0:
                if splitted[0] == 'map_Kd':
                    textureMapPath = shortPath+splitted[1]
                    self.textureMap = cv2.imread(textureMapPath)
                    self.textureMap = cv2.cvtColor( self.textureMap , cv2.COLOR_BGR2RGB)
                    self.textureMap = list(self.textureMap / 255.0)
                    self.texHeight = np.size(self.textureMap, 0)
                    self.texWidth = np.size(self.textureMap, 1)

        mtlFile.close()

    ########################################################################################################################

    def loadSegmentationWeights(self):

        self.vertexLabels = []
        self.vertexWeights = []

        try:

            # labels
            segmentationFile = open(self.folderPath + 'segmentation.txt')

            for line in segmentationFile:
                splitted = line.split()
                if len(splitted) > 0:
                    self.vertexLabels.append(int(splitted[0]))
            segmentationFile.close()
            assert(len(self.vertexLabels) == self.numberOfVertices)

            # weights
            for v in range(0, len(self.vertexLabels)):
                label = self.vertexLabels[v]
                # background / dress / coat / jumpsuit / skirt
                if (label == 0 or label == 6 or label == 7 or label == 10 or label == 12):
                    self.vertexWeights.append(10.0)
                # upper clothes
                elif (label == 5):
                    self.vertexWeights.append(10.0)
                # pants
                elif (label == 9):
                    self.vertexWeights.append(15.0)
                # scarf / socks
                elif (label == 11 or label == 8):
                    self.vertexWeights.append(50.0)
                # skins
                elif (label == 14 or label == 15 or label == 16  or label == 17):
                    self.vertexWeights.append(200.0)
                # shoes / glove / sunglasses / hat
                elif (label == 18 or label == 19 or label == 1 or label == 3 or label == 4):
                    self.vertexWeights.append(200.)
                # hat / hair / face
                elif (label == 2 or label == 13):
                    self.vertexWeights.append(200.0)
                # else
                else:
                    self.vertexWeights.append(200.0)
                    print('Vertex %d has no valid label', v)

        except IOError:
            print("Could not open file! Please close Excel!")
