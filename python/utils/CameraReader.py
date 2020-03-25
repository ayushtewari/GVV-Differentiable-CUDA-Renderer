import numpy as np

class CameraReader:

    def __init__(self, filename, renderResolutionU,renderResolutionV):

        self.filename = filename

        file = open(filename)

        #read faces
        self.extrinsics =[]
        self.intrinsics = []
        self.originalSizeU = []
        self.originalSizeV = []

        for line in file:

            splittedLine = line.split()

            if(splittedLine[0] == 'intrinsic'):
                for i in range(1, len(splittedLine)):
                    if(i % 4 != 0 and i <= 11):
                        self.intrinsics.append(float(splittedLine[i]))

            if (splittedLine[0] == 'extrinsic'):
                for i in range(1, len(splittedLine)):
                    if (i <= 12):
                        self.extrinsics.append(float(splittedLine[i]))

            if (splittedLine[0] == 'size'):
                self.originalSizeU.append(float(splittedLine[1]))
                self.originalSizeV.append(float(splittedLine[2]))

        self.numberOfCameras = int(len(self.extrinsics) / 12)

        self.intrinsics = np.asarray(self.intrinsics)
        self.intrinsics = self.intrinsics.reshape((self.numberOfCameras,3,3))


        for c in range(0,self.numberOfCameras):
            self.intrinsics[c, 0, 0] = (self.intrinsics[c, 0, 0] / self.originalSizeU[c]) * renderResolutionU
            self.intrinsics[c, 1, 1] = (self.intrinsics[c, 1, 1] / self.originalSizeV[c]) * renderResolutionV

            self.intrinsics[c, 0, 2] = (self.intrinsics[c, 0, 2] / self.originalSizeU[c]) * renderResolutionU
            self.intrinsics[c, 1, 2] = (self.intrinsics[c, 1, 2] / self.originalSizeV[c]) * renderResolutionV

        self.intrinsics = self.intrinsics.flatten()
        self.intrinsics= list(self.intrinsics)