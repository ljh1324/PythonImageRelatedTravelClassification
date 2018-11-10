"""
  data/label 폴더 아래에 있는 모든 샘플을 불러와 shuffle 후 train, test를 나눠준다.
"""

import numpy as np
import os, sys
import functools
import defines
from keras.preprocessing import image # image.load_img
import random


def loadAllData(rootfolder, imgSize, labelSize, trainLength, testLength):
  """
  Path 읽기
  0_안녕하세요, 1_바다 ... 안의 각 샘플 폴더 모두 취합
    샘플 폴더 e.g. 1_바다/../2018-05-22_230321_kyg
  
  # arguments
    e.g. rootfolder = ../data/train      
  """

  # remove front back image
  height, width = imgSize

  trainImageList = np.array([], dtype=np.int64).reshape(0, height, width, 3)
  trainLabelList = np.array([], dtype=np.int64).reshape(0, labelSize)

  testImageList = np.array([], dtype=np.int64).reshape(0, height, width, 3)
  testLabelList = np.array([], dtype=np.int64).reshape(0, labelSize)

  label = 0
  for labelFolder in  os.listdir(rootfolder):
      if labelFolder == 'temp':
          continue
      print(labelFolder)
      classDirPath = os.path.join(rootfolder, labelFolder)

      classImageList = loadImageFiles(classDirPath, imgSize)
      classTrainImageList = classImageList[:trainLength]
      classTestImageList = classImageList[trainLength:trainLength + testLength]

      classTrainLabelList = makeLabelList(label, labelSize, trainLength)
      classTestLabelList = makeLabelList(label, labelSize, testLength)

      trainImageList = np.vstack([trainImageList, classTrainImageList])
      trainLabelList = np.vstack([trainLabelList, classTrainLabelList])

      testImageList = np.vstack([testImageList, classTestImageList])
      testLabelList = np.vstack([testLabelList, classTestLabelList])

      label += 1

      #print(trainLabels)
      #print(trainSampleImageList.shape)  

  print('Samples Data Shape:')
  print(trainImageList.shape)
  print(trainLabelList.shape)
  print(testImageList.shape)
  print(testLabelList.shape)

  return trainImageList, trainLabelList, testImageList, testLabelList


""" 이미 정해진 trainLength, testLength를 사용하여 특정 크기로 저장
def loadAllDataAndSaveNumpy(rootfolder, imgSize, labelSize, trainLength, testLength, labelMapper):
  # remove front back image
  height, width = imgSize

  trainImageList = np.array([], dtype=np.int64).reshape(0, height, width, 3)
  trainLabelList = np.array([], dtype=np.int64).reshape(0, labelSize)

  testImageList = np.array([], dtype=np.int64).reshape(0, height, width, 3)
  testLabelList = np.array([], dtype=np.int64).reshape(0, labelSize)

  for labelFolder in  os.listdir(rootfolder):
      if labelFolder == 'temp':
          continue
      print(labelFolder)
      label = labelMapper[labelFolder]
      print(label)

      classDirPath = os.path.join(rootfolder, labelFolder)

      classImageList = loadImageFiles(classDirPath, imgSize)
      classTrainImageList = classImageList[:trainLength]
      classTestImageList = classImageList[trainLength:trainLength + testLength]

      classTrainLabelList = makeLabelList(label, labelSize, trainLength)
      classTestLabelList = makeLabelList(label, labelSize, testLength)

      trainImageList = np.vstack([trainImageList, classTrainImageList])
      trainLabelList = np.vstack([trainLabelList, classTrainLabelList])

      testImageList = np.vstack([testImageList, classTestImageList])
      testLabelList = np.vstack([testLabelList, classTestLabelList])

      label += 1

      #print(trainLabels)
      #print(trainSampleImageList.shape)

  print('Samples Data Shape:')
  print(trainImageList.shape)
  print(trainLabelList.shape)
  print(testImageList.shape)
  print(testLabelList.shape)

  np.save('./numpyData/trainImageList.npy', trainImageList)  # 한글 이미지 데이터 저장 폴더
  np.save('./numpyData/trainLabelList.npy', trainLabelList)
  np.save('./numpyData/testImageList.npy', testImageList)
  np.save('./numpyData/testLabelList.npy', testLabelList)
"""

# 각 클래스마다 80:20의 비율로 저장
def loadAllDataAndSaveNumpy(rootfolder, imgSize, labelSize, labelMapper, labelLength, saveDir):
  """
  Path 읽기
  0_안녕하세요, 1_바다 ... 안의 각 샘플 폴더 모두 취합
    샘플 폴더 e.g. 1_바다/../2018-05-22_230321_kyg
  
  # arguments
    e.g. rootfolder = ../data/train      
  """

  # remove front back image
  height, width = imgSize

  trainImageList = np.array([], dtype=np.int64).reshape(0, height, width, 3)
  trainLabelList = np.array([], dtype=np.int64).reshape(0, labelSize)

  testImageList = np.array([], dtype=np.int64).reshape(0, height, width, 3)
  testLabelList = np.array([], dtype=np.int64).reshape(0, labelSize)

  for labelFolder in  os.listdir(rootfolder):
      if labelFolder == 'temp':
          continue
      print(labelFolder)
      label = labelMapper[labelFolder]
      print(label)

      classDirPath = os.path.join(rootfolder, labelFolder)

      #classImageList = loadImageFiles(classDirPath, imgSize)
      classImageList = loadImageFilesInSize(classDirPath, imgSize, labelLength)
      print(len(classImageList))

      trainLength = int(len(classImageList) * 0.8)
      testLength = len(classImageList) - trainLength

      classTrainImageList = classImageList[:trainLength]
      classTestImageList = classImageList[trainLength:trainLength + testLength]

      classTrainLabelList = makeLabelList(label, labelSize, trainLength)
      classTestLabelList = makeLabelList(label, labelSize, testLength)

      trainImageList = np.vstack([trainImageList, classTrainImageList])
      trainLabelList = np.vstack([trainLabelList, classTrainLabelList])

      testImageList = np.vstack([testImageList, classTestImageList])
      testLabelList = np.vstack([testLabelList, classTestLabelList])

      label += 1
      
      #print(trainLabels)
      #print(trainSampleImageList.shape)

  print('Samples Data Shape:')
  print(trainImageList.shape)
  print(trainLabelList.shape)
  print(testImageList.shape)
  print(testLabelList.shape)

  np.save(saveDir + '/trainImageList.npy', trainImageList) # 영어 데이터 + 한글 데이터 저장폴더
  np.save(saveDir + '/trainLabelList.npy', trainLabelList)
  np.save(saveDir + '/testImageList.npy', testImageList)
  np.save(saveDir + '/testLabelList.npy', testLabelList)


def loadBatchData(rootfolder, imgSize, labelSize, labelLength):
  """
  Path 읽기
  0_안녕하세요, 1_바다 ... 안의 각 샘플 폴더 모두 취합
    샘플 폴더 e.g. 1_바다/../2018-05-22_230321_kyg
  
  # arguments
    e.g. rootfolder = ../data/train      
  """

  # remove front back image
  height, width = imgSize

  trainImageList = np.array([], dtype=np.int64).reshape(0, height, width, 3)
  trainLabelList = np.array([], dtype=np.int64).reshape(0, labelSize)

  label = 0
  for labelFolder in  os.listdir(rootfolder):
      if labelFolder == 'temp':
          continue
      print(labelFolder)
      classDirPath = os.path.join(rootfolder, labelFolder)

      classImageList = loadImageFilesInSize(classDirPath, imgSize, labelLength)
      classLabelList = makeLabelList(label, labelSize, labelLength)
      #classTrainImageList = classImageList[:labelLength]

      #classTrainLabelList = makeLabelList(label, labelSize, labelLength)

      trainImageList = np.vstack([trainImageList, classImageList])
      trainLabelList = np.vstack([trainLabelList, classLabelList])

      label += 1

      #print(trainLabels)
      #print(trainSampleImageList.shape)  

  trainImageList, trainLabelList = shuffleDatasets(trainImageList, trainLabelList)

  #print('Samples Data Shape:')
  #print(trainImageList.shape)
  #print(trainLabelList.shape)

  return trainImageList, trainLabelList


def loadNumpyData(dirPath, files):
  """
  files = [trainImageList, trainLabelList, testImageList, testLabelList]
  """

  # remove front back image
  dataList = []

  for filePath in files:
    data = np.load(os.path.join(dirPath, filePath))
    dataList.append(data)

  return tuple(dataList)


def loadClassData(classDir, imageSize):
  imageList = [] # left, right sum

  # 이미지 로드
  #imageList = _loadImageFiles(dirpath, imageFileList, isShow, imageSize)
  # 시계열 기준으로 이미지를 좌우 손을 불러와 좌우로 붙여서 만들어진 배열을 받아옴
  # imageList = loadImageFilesWithConcat(sampleDir, imageFileList, imageSize)
  imageList = loadImageFiles(classDir, imageSize)
  #imageList = loadImageFilesByHog(sampleDir, imageFileList)
  #imageList = loadImageFilesWithCanny(sampleDir, imageFileList)
  #imageList = loadImageFilesWithDistance(sampleDir, imageFileList, 100)
  print(classDir)
  print(imageList.shape)

  shuffleDataset(imageList)

  trainImageList = imageList[0:160]
  testImageList = imageList[160:200]

  # 이미지 확인법
  #plt.imshow(imageList[0][0] / 255) #settingwindow
  #plt.show() #show

  return trainImageList, testImageList


### LOAD IMAGE FILE
def loadImageFiles(dirpath, imgSize):  # 왼쪽, 오른쪽 이미지를 번갈아 가면서 imgList에 추가
  filePathList = os.listdir(dirpath)

  imgList = []

  for filePath in filePathList:
      #print(filePath)
      # keras가 제공하는 image.load_img를 이용하여 이미지를 불러옴. imgSize는 다음과 같은 튜플 : (img_height, img_width)
      try:
        img = image.load_img(os.path.join(dirpath, filePath), grayscale=False, target_size=imgSize)
        imgArray = image.img_to_array(img)
        #print(imgArray.shape)
        imgList.append(imgArray)

        imageArray = np.fliplr(imgArray) # 이미지 좌우 반전
        imgList.append(imageArray)
        
      except Exception as e:
        print(e)
  
  imgNumpyArray = np.array(imgList)

  shuffleDataset(imgNumpyArray)

  return imgNumpyArray


# 이미지를 불러올 때 labelLength 만큼 무작위로 불러옵니다.
def loadImageFilesInSize(dirpath, imgSize, labelLength):  # 왼쪽, 오른쪽 이미지를 번갈아 가면서 imgList에 추가
  filePathList = os.listdir(dirpath)
  filePathList = shuffleDataset(filePathList)

  imgList = []

  idx = 0
  for filePath in filePathList:
      #print(filePath)
      # keras가 제공하는 image.load_img를 이용하여 이미지를 불러옴. imgSize는 다음과 같은 튜플 : (img_height, img_width)
      try:
        img = image.load_img(os.path.join(dirpath, filePath), grayscale=False, target_size=imgSize)
        imgArray = image.img_to_array(img)
        #print(imgArray.shape)
        imgList.append(imgArray)
        idx += 1
      except Exception as e:
        print(e)
      
      if idx >= labelLength:
        break
  
  imgNumpyArray = np.array(imgList)

  return imgNumpyArray[:labelLength]


### shuffle
def shuffleDataset(d1):
  '''
  # Return
    shuffled data (d1, d2, d3)
  '''

  indexes = list(range(len(d1)))
  random.shuffle(indexes)

  d1_shuffled = [d1[i] for i in indexes]

  return np.array(d1_shuffled)


def shuffleDatasets(d1, d2):
  '''
  # Return
    shuffled data (d1, d2, d3)
  '''

  indexes = list(range(len(d1)))
  random.shuffle(indexes)

  d1_shuffled = [d1[i] for i in indexes]
  d2_shuffled = [d2[i] for i in indexes]

  return np.array(d1_shuffled), np.array(d2_shuffled)


def makeLabelList(label, labelSize, listLength):
  labelList = []
  
  labelOneHot = np.zeros(labelSize)
  labelOneHot[label] = 1

  for i in range(listLength):
    labelList.append(labelOneHot)
  
  return np.array(labelList)


def makeLabelMapper(dirpath):
  label = 0
  files = os.listdir(dirpath)
  labelMapper = {}

  for filePath in files:
    labelMapper[filePath] = label
    label += 1
  
  return labelMapper

def testLoadImage(filename):
  img = image.load_img(filename, grayscale=False, target_size=(256, 256))
  numpyData = np.asarray(img)

  print(img)
  print(numpyData)

if __name__ == '__main__':
  #loadAllData('../data/color-image', defines.IMG_SIZE)
  #loadAllData('..\\data', defines.IMAGE_SIZE, defines.LABEL_SIZE, 160, 40)
  
  #labelMapper = makeLabelMapper('..\\data')
  #print(labelMapper)

  #loadAllDataAndSaveNumpy('..\\data', defines.IMAGE_SIZE, defines.LABEL_SIZE, 160, 40, labelMapper) // 클래스마다 정해진 길이로 나누어 저장
  #labelMapper = makeLabelMapper('..\\data_merge')
  #print(labelMapper)
  #loadAllDataAndSaveNumpy('..\\data_merge', defines.IMAGE_SIZE, defines.LABEL_SIZE, labelMapper)

  # saveDir = numpyExpandData
  #labelLength = 1000
  #labelMapper = makeLabelMapper('..\\data_merge_expaned_1000')
  #print(labelMapper)
  #loadAllDataAndSaveNumpy('..\\data_merge_expaned_1000', defines.IMAGE_SIZE, defines.LABEL_SIZE, labelMapper, labelLength)

  #saveDir = 'numpyTenClass'
  #labelLength = 1000
  #labelMapper = makeLabelMapper('..\\data_merge_few_class')
  #print(labelMapper)
  #loadAllDataAndSaveNumpy('..\\data_merge_few_class', defines.IMAGE_SIZE, defines.LABEL_SIZE, labelMapper, labelLength, saveDir)

  saveDir = './numpydata/numpyTenClass3000'
  labelLength = 3000
  labelMapper = makeLabelMapper('..\\data_merge_few_class_3000')
  print(labelMapper)
  loadAllDataAndSaveNumpy('..\\data_merge_few_class_3000', defines.IMAGE_SIZE, defines.LABEL_SIZE, labelMapper, labelLength, saveDir)

  #testLoadImage('test.jpg')