import numpy as np

class CameraReader:

    def __init__(self, filename):

        self.filename = filename

        file = open(filename)

        #read faces
        self.extrinsics =[]
        self.intrinsics = []


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

