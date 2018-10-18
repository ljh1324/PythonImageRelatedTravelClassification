import time
import matplotlib.pyplot as plt
import os

def saveLabData(accLists, lossLists, timeList, fileName, accImgFileName, lossImgFileName):
    '''
    now = time.localtime()
    s = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)
    if not os.path.exists(s):
        os.makedirs(s)
    '''

    trainAcc, testAcc = accLists
    trainLoss, testLoss = lossLists

    plt.cla()
    plt.plot(trainAcc, 'r', label='train_acc')
    plt.plot(testAcc, 'r--', label='test_acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    #plt.show()
    plt.savefig(accImgFileName)

    plt.cla()
    plt.title('loss')
    plt.plot(trainLoss, 'r', label='train_loss')
    plt.plot(testLoss, 'r--', label='test_loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    #plt.show()
    plt.savefig(lossImgFileName)

    f = open(fileName, 'w')

    length = len(trainAcc)

    f.write('Train Acc\n')
    for i in range(length - 1):
        f.write('{0},'.format(trainAcc[i]))
    f.write('{0}\n\n'.format(trainAcc[length - 1]))
    
    f.write('Train Max Acc\n')
    f.write('{0}\n\n'.format(max(trainAcc)))

    f.write('Test Acc\n')
    for i in range(length - 1):
        f.write('{0},'.format(testAcc[i]))
    f.write('{0}\n\n'.format(testAcc[length - 1]))

    f.write('Test Max Acc\n')
    f.write('{0}\n\n'.format(max(testAcc)))

    f.write('Train Loss\n')
    for i in range(length - 1):
        f.write('{0},'.format(trainLoss[i]))
    f.write('{0}\n\n'.format(trainLoss[length - 1]))

    f.write('Train Min Loss\n')
    f.write('{0}\n\n'.format(min(trainLoss)))

    f.write('Test Loss\n')
    for i in range(length - 1):
        f.write('{0},'.format(testLoss[i]))
    f.write('{0}\n\n'.format(testLoss[length - 1]))

    f.write('Test Min Loss\n')
    f.write('{0}\n\n'.format(min(testLoss)))

    f.write('Time per epoch\n')
    for i in range(length - 1):
        f.write('{0},'.format(timeList[i]))
    f.write('{0}\n\n'.format(timeList[length - 1]))

    f.write('Total Time\n')
    f.write('{0}\n'.format(sum(timeList)))

    f.close()
