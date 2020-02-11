

class OBJReader:

    def __init__(self, filename):

        self.filename = filename


        file = open(filename)

        #read faces
        self.facesVertexId =[]
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
