import cv2
import numpy as np
import os
from SudokoSolver import init_hog_descripter,hog_compute

def get_train_data(path):
    files = os.listdir(path)
    trains = []
    labels = []
    hog = init_hog_descripter()
    for file in files:
        image = cv2.imread(path + '/' + file)
        hist = hog_compute(hog, image)
        hist = np.reshape(hist, (-1))
        trains.append(hist)
        labels.append((int(file.split("-")[0])))
    print(hist.shape)
    trains = np.matrix(trains, dtype=np.float32)
    labels = np.array(labels)
    labels.resize((labels.shape[0], 1))

    return trains, labels


def train_svm_model(data_path):
    trains, labels = get_train_data(data_path)
    saved = False
    if not saved:
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        # svm.setC(2.67)
        # svm.setGamma(5.383)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
        # tr = np.matrix([t for t in trains],dtype=np.float32)
        svm.train(trains, cv2.ml.ROW_SAMPLE, labels)
        svm.save('svm_model.xml')
    else:
        svm = cv2.ml.SVM_load('svm_model.xml')
    print("Train finished")
    return svm


train_svm_model('train-data')
