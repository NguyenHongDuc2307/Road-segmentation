from datetime import datetime
from PIL import Image
import sys
import pandas as pd 
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputLayer, Dense 
from PIL import Image

rectSize = 5;

inputImagePath = 'image-input'
inputImageFile = sys.argv[1]
inputImage = Image.open(inputImagePath + '/' + inputImageFile)
inputImageXSize, inputImageYSize = inputImage.size

outputImagePath = 'image-output'
outputImageFile = sys.argv[2]
outputImage = inputImage.crop((rectSize//2, rectSize//2, inputImageXSize - (rectSize//2), inputImageYSize - (rectSize//2)))
outputImageXSize, outputImageYSize = outputImage.size

print(str(datetime.now()) + ': initializing model...')

def extractFeatures():
    features = np.zeros((((inputImageXSize - ((rectSize//2)*2)) * (inputImageYSize - ((rectSize//2)*2)))+1, rectSize*rectSize*3), dtype=np.int)
    rowIndex = 0
    for x in range(rectSize//2, inputImageXSize - (rectSize//2)):
        for y in range(rectSize//2, inputImageYSize - (rectSize//2)):            
            rect = (x - (rectSize//2), y - (rectSize//2), x + (rectSize//2) + 1, y + (rectSize//2) + 1)
            subImage = inputImage.crop(rect).load()
            colIndex = 0
            for i in range(rectSize):
                for j in range(rectSize):
                    features[rowIndex, colIndex] = subImage[i, j][0]
                    colIndex += 1
                    features[rowIndex, colIndex] = subImage[i, j][1]
                    colIndex += 1
                    features[rowIndex, colIndex] = subImage[i, j][2]
                    colIndex += 1            
            rowIndex += 1    
    return features

def constructOutputImage(prediction):
    outputImagePixels = outputImage.load()
    rowIndex = 0
    for x in range(outputImageXSize):
        for y in range(outputImageYSize):
            outputImagePixels[x, y] = ((255, 255, 255) if prediction[rowIndex] else (0, 0, 0))
            rowIndex += 1
    

print(str(datetime.now()) + ': processing image', inputImageFile)
features = extractFeatures()
validData = pd.DataFrame(features, columns=np.arange(1,rectSize*rectSize*3+1))

for i in validData.columns[0:]:
    data_min = 0
    data_max = 255
    validData[i] = (validData[i] - data_min) / (data_max - data_min)


#validData.to_csv("data" + inputImageFile.split(".")[0]+ ".csv", index=False)
#print(str(datetime.now()) + ': done')

print(str(datetime.now()) + ': initializing model...')
model = keras.models.load_model('model')

print(str(datetime.now()) + ': processing image', inputImageFile)
prediction = model.predict_classes(validData)


print(str(datetime.now()) + ': constructing output image...')
constructOutputImage(prediction)

print(str(datetime.now()) + ': saving output image...')
outputImage.save(outputImagePath + '/' + outputImageFile, 'png')

print(str(datetime.now()) + ': done')

