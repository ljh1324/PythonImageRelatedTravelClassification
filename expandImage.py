from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import math

def expandAllLabelData(rootDir, saveDir, numOfClassImage):
  for labelFolder in  os.listdir(rootDir):
    if labelFolder == 'temp':
      continue
    classDirPath = rootDir + '/' + labelFolder
    
    filePathList = getFilePathListInDir(classDirPath)
    saveClassDir = saveDir + '/' + labelFolder
    
    print(filePathList)

    if not os.path.exists(saveClassDir):
      os.makedirs(saveClassDir)
    
    expandImageList(filePathList, saveClassDir, numOfClassImage)
    

def getFilePathListInDir(dirPath):
  fileList = os.listdir(dirPath)

  filePathList = []
  print(dirPath)
  for filePath in fileList:
    try:
      img = load_img(dirPath + '/' + filePath)  # 이미지가 로드 되는지 테스트
      filePathList.append((dirPath, filePath))  # ('디렉토리명', '파일명')
    except Exception as e:
      print(e)

  return filePathList


def expandImageList(filePathList, saveDir, numOfImage):
  needImage = math.ceil(numOfImage / len(filePathList)) # numOfImage만큼 이미지를 만들기 위해 각 이미지마다 필요한 이미지 개수

  for filePath in filePathList:
    expandImage(filePath[0], filePath[1], saveDir, needImage)


def expandImage(originDir, filename, saveDir, numOfImage):
  filepath = originDir + '/' + filename

  datagen = ImageDataGenerator(
    rotation_range=3,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

  img = load_img(filepath)  # this is a PIL image
  x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
  x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

  # the .flow() command below generates batches of randomly transformed images
  # and saves the results to the `preview/` directory
  i = 0
  for batch in datagen.flow(x, batch_size=1, save_to_dir=saveDir, save_prefix=filename, save_format='jpeg'):
    i += 1
    if i > numOfImage:
      break  # otherwise the generator would loop indefinitely
  

if __name__ == '__main__':
  expandAllLabelData('../data_merge', '../data_merge_expaned', 10000)