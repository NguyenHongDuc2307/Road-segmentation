import random
from os import listdir
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime

InputImagesPath = 'airs-dataset/input'
OutputImagesPath = 'airs-dataset/output'
InputImagesFiles = listdir(InputImagesPath)
OutputImagesFiles = listdir(OutputImagesPath)
rectSize = 5
column = rectSize*rectSize*3+1


print(str(datetime.now()) + ': InputImagesFiles:', len(InputImagesFiles))
print(str(datetime.now()) + ': OutputImagesFiles:',  len(OutputImagesFiles))
if(len(InputImagesFiles) != len(OutputImagesFiles)):
    raise Exception('input images and output images number mismatch')

for i in range(len(InputImagesFiles)):
    inputImageFile = InputImagesFiles[i][:-5]
    outputImageFile = OutputImagesFiles[i][:-5]
    if(inputImageFile != outputImageFile):
        raise Exception('inputImageFile and outputImageFile mismatch at index', str(i))

print(str(datetime.now()) + ': input and output files check success')


def writeDataFile(inputImagePath, outputImagePath, inputImageFiles, outputImageFiles, dataFileName):
    dataFile = open(dataFileName, 'w')
    linesCount = 0
    linesLimit = 2000000
    linesCountPerImage = 0
    linesLimitPerImage = (linesLimit / len(inputImageFiles)) + 1
    

    headerList = range(1, column+1)
    headerString = ''.join([str(elem)+',' for elem in headerList])
    headerString = headerString[0:len(headerString)-1]
    line = ''
    line += headerString + '\n'
    linesCount += 1
    dataFile.write(line)
    
    for i in range(len(inputImageFiles)):
        print(str(datetime.now()) + ': processing image', i)
        linesCountPerImage = 0
        inputImage = Image.open(inputImagePath + '/' + inputImageFiles[i])
        inputImageXSize, inputImageYSize = inputImage.size
        # inputImagePixels = inputImage.load()
        
        outputImage = Image.open(outputImagePath + '/' + outputImageFiles[i])
        outputImageXSize, outputImageYSize = outputImage.size
        outputImagePixels = outputImage.load()
        
        if((inputImageXSize != outputImageXSize) or (inputImageYSize != outputImageYSize)):
            raise Exception('inputImage and outputImage mismatch at index', str(i))

        outputImageRoadPixelsArr = [];
        outputImageNonRoadPixelsArr= [];
        
        for x in range(rectSize//2, inputImageXSize - (rectSize//2)):
            for y in range(rectSize//2, inputImageYSize - (rectSize//2)):
                isRoadPixel = outputImagePixels[x, y]
                if(isRoadPixel):
                    outputImageRoadPixelsArr.append((x, y))
                else:
                    outputImageNonRoadPixelsArr.append((x, y))

        random.shuffle(outputImageRoadPixelsArr)
        random.shuffle(outputImageNonRoadPixelsArr)

        print(len(outputImageRoadPixelsArr))
        print(len(outputImageNonRoadPixelsArr))
        
        for m in range(len(outputImageRoadPixelsArr)):
            if(linesCountPerImage >= linesLimitPerImage):
                break
            
            if(((m*2) + 1) >= len(outputImageNonRoadPixelsArr)):
                break
            
            x = outputImageRoadPixelsArr[m][0];
            y = outputImageRoadPixelsArr[m][1];
            
            rect = (x - (rectSize//2), y - (rectSize//2), x + (rectSize//2) + 1, y + (rectSize//2) + 1)
            subImage = inputImage.crop(rect).load()
            line = ''
            for i in range(rectSize):
                for j in range(rectSize):
                    line += str(subImage[i, j][0]) + ','
                    line += str(subImage[i, j][1]) + ','
                    line += str(subImage[i, j][2]) + ','
            
            line += str(1) + '\n'
            linesCount += 1
            linesCountPerImage += 1
            dataFile.write(line)
            
            for n in range(2):
                x = outputImageNonRoadPixelsArr[(m*2) + n][0];
                y = outputImageNonRoadPixelsArr[(m*2) + n][1];
                
                rect = (x - (rectSize//2), y - (rectSize//2), x + (rectSize//2) + 1, y + (rectSize//2) + 1)
                subImage = inputImage.crop(rect).load()
                line = ''
                for i in range(rectSize):
                    for j in range(rectSize):
                        line += str(subImage[i, j][0]) + ','
                        line += str(subImage[i, j][1]) + ','
                        line += str(subImage[i, j][2]) + ','
                
                line += str(0) + '\n'
                linesCount += 1
                linesCountPerImage += 1
                dataFile.write(line)
    
    print(str(datetime.now()) + ': ' + dataFileName + ' linesCount:', linesCount)

DataFileName = 'airs-dataset/data.csv'


print(str(datetime.now()) + ': writing DataFile')
writeDataFile(InputImagesPath, OutputImagesPath, InputImagesFiles, OutputImagesFiles, DataFileName)


print(str(datetime.now()) + ': Data preprocessing')
data = pd.read_csv('airs-dataset/data.csv')
data[str(column)].fillna(0 , inplace=True)

for i in range(1,column):
  data[str(i)].fillna(data[str(i)].mean(), inplace=True)


for i in data.columns[0:column-1]:
    data_max = 255
    data[i] = data[i]/data_max


data.to_csv(DataFileName, index=False)
print(str(datetime.now()) + ': DataFile complete')