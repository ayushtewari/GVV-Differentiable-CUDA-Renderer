
import cv2

class OBJReader:

    def __init__(self, filename):

        self.filename = filename

        file = open(filename)

        #read faces
        self.facesVertexId =[]
        self.facesTextureId = []
        self.vertexColors = []
        self.vertexCoordinates = []
        self.pertVertexTextureCoordinate = []

        for line in file:
            splitted = line.split()
            if len(splitted) >0:

                #is face
                if splitted[0] == 'f':
                    v0 = splitted[1].split('/')[0]
                    v1 = splitted[2].split('/')[0]
                    v2 = splitted[3].split('/')[0]

                    self.facesVertexId.append(int(v0)-1)
                    self.facesVertexId.append(int(v1)-1)
                    self.facesVertexId.append(int(v2)-1)

                    t0 = splitted[1].split('/')[1]
                    t1 = splitted[2].split('/')[1]
                    t2 = splitted[3].split('/')[1]

                    self.facesTextureId.append(int(t0) - 1)
                    self.facesTextureId.append(int(t1) - 1)
                    self.facesTextureId.append(int(t2) - 1)

                # is vertex
                if splitted[0] == 'v':
                    self.vertexCoordinates.append([float(splitted[1]),float(splitted[2]),float(splitted[3])])
                    self.vertexColors.append([float(splitted[4]), float(splitted[5]), float(splitted[6])])

                # is texture coordinate
                if splitted[0] == 'vt':
                    self.pertVertexTextureCoordinate.append([float(splitted[1]), float(splitted[2])])

                # is mtllib
                if splitted[0] == 'mtllib':
                    mtlFilePath = filename[0:filename.rfind('/')+1]

                    mtlName = splitted[1][2:]

                    mtlFilePathFull = mtlFilePath + mtlName

                    self.loadMtlTexture(mtlFilePathFull,mtlFilePath)

        self.textureCoordinates = []
        for t in range(0,len(self.facesTextureId)):
            texCoord = self.pertVertexTextureCoordinate[self.facesTextureId[t]]
            self.textureCoordinates.append(texCoord[0])
            self.textureCoordinates.append(texCoord[1])


    def loadMtlTexture(self,mtlFileName,shortPath):
        mtlFile = open(mtlFileName)
        for line in mtlFile:
            splitted = line.split()
            if len(splitted) > 0:
                if splitted[0] == 'map_Kd':
                    textureMapPath = shortPath+splitted[1]

                    self.textureMap = cv2.imread(textureMapPath)
                    self.textureMap = cv2.cvtColor( self.textureMap , cv2.COLOR_BGR2RGB)
                    self.textureMap =  self.textureMap /255.0