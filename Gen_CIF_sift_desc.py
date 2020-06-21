import cv2
import numpy as np
import csv
import pickle


def load_cfar10_batch(path):
    with open(path, mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

sift_extractor = cv2.xfeatures2d.SIFT_create()
des_list = np.empty(shape=[0,128])#descriptor length of surf
image_des_len = []
features = []
labels=[]

print("start")
images,class_labels=load_cfar10_batch("/media/vidhikatkoria/Research/VR/Assignment2/cifar-10-python/cifar-10-batches-py/data_batch_1")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        des_list = np.concatenate([des_list, descriptor])
        image_des_len.append(len(descriptor))
    else:
        index += [id]
    id += 1
    print(id)
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=class_labels
print("1------------")

images,class_labels=load_cfar10_batch("/media/vidhikatkoria/Research/VR/Assignment2/cifar-10-python/cifar-10-batches-py/data_batch_2")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        des_list = np.concatenate([des_list, descriptor])
        image_des_len.append(len(descriptor))
    else:
        index += [id]
    id += 1
    print(id)
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=labels+class_labels
print("2------------")

images,class_labels=load_cfar10_batch("/media/vidhikatkoria/Research/VR/Assignment2/cifar-10-python/cifar-10-batches-py/data_batch_3")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        des_list = np.concatenate([des_list, descriptor])
        image_des_len.append(len(descriptor))
    else:
        index += [id]
    id += 1
    print(id)
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=labels+class_labels
print("3------------")

images,class_labels=load_cfar10_batch("/media/vidhikatkoria/Research/VR/Assignment2/cifar-10-python/cifar-10-batches-py/data_batch_4")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        des_list = np.concatenate([des_list, descriptor])
        image_des_len.append(len(descriptor))
    else:
        index += [id]
    id += 1
    print(id)
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=labels+class_labels
print("4------------")

images,class_labels=load_cfar10_batch("/media/vidhikatkoria/Research/VR/Assignment2/cifar-10-python/cifar-10-batches-py/data_batch_5")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        des_list = np.concatenate([des_list, descriptor])
        image_des_len.append(len(descriptor))
    else:
        index += [id]
    id += 1
    print(id)
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=labels+class_labels
print("5------------")

images,class_labels=load_cfar10_batch("/media/vidhikatkoria/Research/VR/Assignment2/cifar-10-python/cifar-10-batches-py/test_batch")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        des_list = np.concatenate([des_list, descriptor])
        image_des_len.append(len(descriptor))
    else:
        index += [id]
    id += 1
    print(id)
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=labels+class_labels
print("test------------")

with open('CIFAR_SIFT_DESC.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(des_list)

with open('CIFAR_SIFT_Label_Len.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows([labels,image_des_len])




