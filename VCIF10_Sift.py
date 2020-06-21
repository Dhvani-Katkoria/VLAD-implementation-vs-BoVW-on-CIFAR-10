import cv2
import numpy as np
import csv
import pickle


des_list=[]
for row in csv.reader(open('CIFAR_SIFT_DESC.csv', 'r'), delimiter=','):
     des_list.append([float(i) for i in row])

L=[]
for row in csv.reader(open('CIFAR_SIFT_Label_Len.csv', 'r'), delimiter=','):
     L.append([int(i) for i in row])
[labels,image_des_len]=L


print("computing clusters........")
#kmeans clustering
des_list = np.float32(des_list)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 32
ret, label, centers = cv2.kmeans(des_list, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)#clusters and iterations
print("computing features.........")
init_id = 0
features = []
for img_des_id in image_des_len:
    descrips = des_list[init_id: init_id + img_des_id]
    label_des = label[init_id: init_id + img_des_id]#labels of each descriptor
    center_des = centers[label_des]#nearest center point descriptors for each descriptor
    vlad = np.zeros(shape=[K, 128])
    for i in range(img_des_id):#for each descriptor vec of image
        vlad[label_des[i]] = vlad[label_des[i]] + descrips[i] - center_des[i]
    init_id += img_des_id

    #vlad contains aggregated descriptor for each cluster of image
    vlad_norm = vlad.copy()
    cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
    X=vlad_norm.reshape(K * 128, -1).T
    features.append((X).tolist()[0])#flatten the obtained matrix for each image and add it base

features+=[labels]
with open('CIFAR_VLAD_SIFT_FEATURES.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(features)

