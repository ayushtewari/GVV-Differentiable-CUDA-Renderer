
########################################################################################################################
# Imports
########################################################################################################################

import subprocess
from sys import platform
import pandas as pd
from io import StringIO
import os
import tensorflow as tf

########################################################################################################################
# Check free GPU
########################################################################################################################

def get_free_gpu():

    if platform == "linux" or platform == "linux2":
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    elif platform == "win32" or platform == "win64":
        gpu_stats = subprocess.check_output(["nvidia-smi.exe", "--format=csv", "--query-gpu=memory.used,memory.free"])   #assumes program runs in C drive

    gpu_stats = gpu_stats.decode('ascii')
    stringIoGPUStats = StringIO(gpu_stats)
    gpu_df = pd.read_csv(stringIoGPUStats,names=['memory.used', 'memory.free'],skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))

    freeGPUMemoryArray = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))._values
    numberOfGPUs = freeGPUMemoryArray.shape[0]

    largestFreeMemory = -1
    selectedGPU = -1

    for gpu in range(0,numberOfGPUs):
        if(int(freeGPUMemoryArray[gpu]) > largestFreeMemory):
            largestFreeMemory =int(freeGPUMemoryArray[gpu])
            selectedGPU = gpu

    if(largestFreeMemory < 5000):
        selectedGPU = -1

    if (selectedGPU == -1):

        print('No free GPU available')
        return False

    else:
        print('--Mask all other GPUS')
        freeGPUIdString = str(selectedGPU)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = freeGPUIdString

        print('--Use GPU ' + str(selectedGPU))
        print('Tensorflow Version:  ' + tf.__version__)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        return True

########################################################################################################################
#
########################################################################################################################

