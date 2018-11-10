import time
import datetime

import interfaceUtils as util

util.showProcess('Start Program')

import loadData as ld
import models as models
import train as train
import defines as defines
import saveLabData as savelab

import tensorflow as tf
from keras import backend as K

util.showProcess('Loading TF Session')
sess = tf.Session()
K.set_session(sess)

# configs
epochs = util.inputInt('How many epochs?')
batch_size = 32 # GPU 메모리 부족으로 Batch_size에 한계 있음

isLoadWeight = util.inputInt('Load weight? (1 to yes)')
overwrite = True

dataPath = '../data'
weightPath = './weights'
modelImagePath = './modelImgs'

imageSize = defines.IMAGE_SIZE
labelSize = defines.LABEL_SIZE
#trainLength = 160
#testLength = 40

# load dataset
util.showProcess('Loading dataset')

print('Loading Samples : ')
#trainImageList, trainLabelList, testImageList, testLabelList = \
#    ld.loadAllData(dataPath, imageSize, labelSize, trainLength, testLength) # image
#dataFolder = './numpyTenClass'
#files = ['trainImageList.npy', 'trainLabelList.npy', 'testImageList.npy', 'testLabelList.npy']
#trainImageList, trainLabelList, testImageList, testLabelList = ld.loadNumpyData(dataFolder, files)

batchFolder = '../data_merge_few_class_3000'
testsetFolder = '../data_merge_few_class_origin'
testImageList, testLabelList = ld.loadBatchData(testsetFolder, defines.IMAGE_SIZE, defines.LABEL_SIZE, 100)
# make model, b1/b2는 구조 print용
util.showProcess('Model Generating')

#m1 = models.model(imageSize, labelSize)
m1 = models.vgg_model(imageSize, labelSize)

models.saveModelDescription(m1, modelImagePath, False)

# Load Weight
if isLoadWeight == 1:
    util.showProcess('Loading Weight')
    models.loadWeight(m1, weightPath)

# Train
util.showProcess('Train M1')
#accList = []

trainAcc = []
trainLoss = []

testAcc = []
testLoss = []

totalTime = 0
timeList = []

for i in range(epochs):
  startTime = time.time()
  trainImageList, trainLabelList = ld.loadBatchData(batchFolder, defines.IMAGE_SIZE, defines.LABEL_SIZE, 10)
  hist = m1.fit(trainImageList, trainLabelList,
    epochs=1,
    verbose=1,
    batch_size=batch_size)
  print(hist.history)

  trainAcc.append(hist.history['acc'][0])
  trainLoss.append(hist.history['loss'][0])

  endTime = time.time()
  timeList.append(endTime - startTime)
  
  '''
  accuracy = train.calculateAccuracy(testImageList,
                                testLabelList,
                                len(testLabelList),
                                m1, verbose=1,
                                batch_size=1)
  '''
  loss, accuracy = m1.evaluate(testImageList, testLabelList)
  testAcc.append(accuracy)
  testLoss.append(loss)

modelName = m1.name
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
fileName = modelName + '_epochs_' + str(epochs) + '_' + st + '.txt'
resultFileName = modelName + '_epochs_' + str(epochs) + '_' + st + '_prediction.txt'
accImgFileName = modelName + '_' + st + '_acc.jpg'
lossImgFileName = modelName + '_' + st + '_loss.jpg'

fileName = 'labdata/' + fileName
accImgFileName = 'labdata/' + accImgFileName
lossImgFileName = 'labdata/' + lossImgFileName
resultFileName = 'labdata/' + resultFileName

savelab.saveLabData((trainAcc, testAcc), (trainLoss, testLoss), timeList, fileName, accImgFileName, lossImgFileName)
try:
    savelab.savePredictionResult(testImageList, testLabelList, m1, resultFileName)
except Exception as e:
    print(e)
print('Accuracy List: ')
print(testAcc)

print('Calculate Accuracy')
accuracy = train.calculateAccuracy(testImageList,
                                testLabelList,
                                len(testLabelList),
                                m1, verbose=1,
                                batch_size=1)

util.showProcess('Evaluate M1 Batch_Size_1')
score, acc = m1.evaluate(trainImageList, trainLabelList, batch_size=1)

print('Test score:', score)
print('Test accuracy:', acc)

util.showProcess('Evaluate M1 Batch_Size')
score, acc = m1.evaluate(trainImageList, trainLabelList, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

# Test
util.showProcess('Test M1')
# def calculateAccuracy(dataList, labelList, labelSize, model, verbose, batch_size):
accuracy = train.calculateAccuracy(testImageList,
                                   testLabelList,
                                   len(testLabelList),
                                   m1, verbose=1,
                                   batch_size=1)
print('Accuracy: ' + str(accuracy))

# Write Weight
if overwrite:
    util.showProcess('Saving Weight')
    models.saveWeight(m1, weightPath)

sess.close()
