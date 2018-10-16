'''
 손가락 예측 모델
 CNN + LSTM
'''
import interfaceUtils as util

util.showProcess('Start Program')

import loadData as ld
import models as models
import train as train
import defines as defines

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
trainLength = 160
testLength = 40

# load dataset
util.showProcess('Loading dataset')

print('Loading Samples : ')
#trainImageList, trainLabelList, testImageList, testLabelList = \
#    ld.loadAllData(dataPath, imageSize, labelSize, trainLength, testLength) # image
dataFolder = './numpyData'
files = ['trainImageList.npy', 'trainLabelList.npy', 'testImageList.npy', 'testLabelList.npy']
trainImageList, trainLabelList, testImageList, testLabelList = ld.loadNumpyData(dataFolder, files)


# make model, b1/b2는 구조 print용
util.showProcess('Model Generating')

m1 = models.model(imageSize, labelSize)

models.saveModelDescription(m1, modelImagePath, False)

# Load Weight
if isLoadWeight == 1:
    util.showProcess('Loading Weight')
    models.loadWeight(m1, weightPath)

# Train
util.showProcess('Train M1')
accList = []
for i in range(epochs):
    m1.fit(trainImageList, trainLabelList,
        epochs=1,
        verbose=1,
        batch_size=batch_size)
    
    accuracy = train.calculateAccuracy(testImageList,
                                   testLabelList,
                                   len(testLabelList),
                                   m1, verbose=1,
                                   batch_size=1)
    
    accList.append(accuracy)
print('Accuracy List: ')
print(accList)

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
