#!/usr/bin/env python
import cv2
import matplotlib.pyplot as plt
import cvlib
from cvlib.object_detection import draw_bbox
from statistics import mean
import sys
import numpy as np
import math
import imagehash
#from PIL import Image
import xlsxwriter

def generateYOLOexcelsheet(all_image_confidence_scores, data_size, model_type):
    workbook = xlsxwriter.Workbook('/home/mengmei/Research/gitlab/hisa_trustzone/HybridTEE/results_log/excelsheets/'+data_size+'/Yolo_'+model_type+'.xlsx')
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    worksheet.write('A1', 'Image', bold)
    for i in range(8):
        worksheet.write(0, i+1, 'L'+str(i), bold)
    image_list = ['Eagle', 'Dog', 'Cat', 'Horses', 'Giraffe']
    for i in range(5):
        worksheet.write(i+1, 0, image_list[i])
    for i in range(5):
        for j in range(8):
            worksheet.write(i+1, j+1, all_image_confidence_scores[i][j])
    workbook.close()


def generateSIFTexcelsheet(total_sift_dist, total_features, data_size, model_type):
    workbook = xlsxwriter.Workbook('/home/mengmei/Research/gitlab/hisa_trustzone/HybridTEE/results_log/excelsheets/'+data_size+'/SIFT_'+model_type+'.xlsx')
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    worksheet.write('A1', 'Image', bold)
    for i in range(8):
        worksheet.write(0, i+1, 'L'+str(i), bold)
    image_list = ['Eagle', 'Dog', 'Cat', 'Horses', 'Giraffe']
    for i in range(5):
        worksheet.write(i+1, 0, image_list[i])
    for i in range(5):
        for j in range(8):
            if total_sift_dist[i][j] == 0:
                sift_value = 0
            else:
                sift_value = total_sift_dist[i][j]/total_features[i]*100
            worksheet.write(i+1, j+1, sift_value)
    workbook.close()


def sift_sim(path_a, path_b):

    # get the images
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    img_a = img_c = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    img_a = cv2.resize(img_a, (img_b.shape[0], img_b.shape[1]))

    # find the keypoints and descriptors with SIFT
    # Initiate SIFT detector
#sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_a,None)
    kp2, des2 = sift.detectAndCompute(img_b,None)
    kp3, des3 = sift.detectAndCompute(img_c,None) #get the original features
    #print('kp1, kp2: ', len(kp1), len(kp2))
    if( des1 is None or des2 is None ):
        return [], len(kp3)
    #print('des1, des2: ', len(des1), len(des2))
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    #print('matches:',len(matches))
    # Apply ratio test
    good = []
    for i in range(0, len(matches)):
        if(len(matches[i]) < 2):
            continue
        else:
            m, n = matches[i]
            if m.distance < 0.75*n.distance:
                good.append(m.distance)
    return good, len(kp3)

#img = cv2.imread('car1.jpg',1)
#bbox, label, conf = cvlib.detect_common_objects(img)
#print(label)

#dict_object = {0:"horse", 1:"elephant", 2:"cow", 3:"motorcycle", 4:"car"}
dict_object = {0:"bird", 1:"dog", 2:"cat", 3:"horse", 4:"giraffe"}
total_confidence_score = []
total_features = []
total_sift_dist = []
layers = [i for i in range(0,8)]
total_confidence_labels = []
all_image_confidence_max_score = []
for k in range(0,5):
    all_image_confidence_max_score.append([])
    input_data = sys.argv[k+1]
    #org_img = cv2.imread(input_data)
    #org_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

    confidence_score = []
    #norm_coef = []
    confidence_label = []
    sift_dist_score = []
    #total_features.append([])
    #layer_feature_list = []
    for i in range(0,8):
        score = 0
        if(i == 0 or i == 1):
            length = 32
        elif(i == 2 or i == 3 or i == 5):
            length = 64
        else:
            length = 128

        conf_score = []
        sift_score = []
        count = 0
        #dist_ncc = []
        channel_feature_list = []
        for j in range(0, length):
            input_img = input_data
            input_img = input_data.replace('.jpg','')
            input_img = input_img + '_'
            input_img = input_img + str(i) + '_' + str(j) + '.jpg'
            #print("Detect object: ",input_img)
            img = cv2.imread(input_img,1)
            bbox, label, conf = cvlib.detect_common_objects(img)
            
            if len(label) > 1:
                label = label[0]
            if label:
                expected_label = dict_object[k]
                if(expected_label == label[0]):
                    count += 1
                    conf_score.append(conf[0])
                #conf_label.append(label)

            good, feature = sift_sim(input_data, input_img)
            #print('feature: ', feature)
            #channel_feature_list.append(feature)
            #print("SIFT good match")
            #print(len(good))
            #print(good)
            sift_score.append(len(good))
            #dist_ncc.append(sum( (org_gray - mean(org_gray)) * (img - mean(img)) ) / ((img.size - 1) * np.std(img) * np.std(org_gray)))

        confidence_score.append(conf_score)
        confidence_label.append(count)
        sift_dist_score.append(max(sift_score))
        all_image_confidence_max_score[-1].append(max(conf_score) if conf_score else 0)
        #print('****',all_image_confidence_max_score)
        #layer_feature_list.append(max(channel_feature_list))
    total_features.append(feature)
    print("Confidence scores")
    print(confidence_score)
    print("Confidence labels")
    print(confidence_label)
    print("SIFT score")
    print(sift_dist_score)
    print("\n")
    #print('dist_score', sift_dist_score[0])

    # Maximum confidence score
    final_score = []
    for conf_scr in confidence_score:
        if not conf_scr:
            final_score.append(0)
        else:
            final_score.append(max(conf_scr))
        #final_score.append(sum(conf_scr)/len(conf_scr))

    print("Final scores")
    print(final_score)
    total_confidence_score.append(final_score)
    total_sift_dist.append(sift_dist_score)
    #total_ncc_dist.append(norm_coef)
    total_confidence_labels.append(confidence_label)
    print('SIFT: ', sift_dist_score)
    print('Total Features: ', total_features[-1])
    print("\n")

print("Total labels detected")
print(total_confidence_labels)
print("Final SIFT score")
print(total_sift_dist)
print("Final #Features")
print(total_features)
print('Final Confidence score')
print(all_image_confidence_max_score)
generateYOLOexcelsheet(all_image_confidence_max_score, sys.argv[6], sys.argv[7])
generateSIFTexcelsheet(total_sift_dist, total_features, sys.argv[6], sys.argv[7])
"""
plt.title("SIFT features matched - Darknet")
plt.ylabel("Percentage match")
plt.xlabel("DNN Layers")
#plt.style.use(['dark_background', 'presentation'])
plt.plot(layers, total_sift_dist[0], 'bo--',linewidth=2, markersize=12)
plt.plot(layers, total_sift_dist[1], 'go--',linewidth=2, markersize=12)
plt.plot(layers, total_sift_dist[2], 'ro--',linewidth=2, markersize=12)
plt.plot(layers, total_sift_dist[3], 'co--',linewidth=2, markersize=12)
plt.plot(layers, total_sift_dist[4], 'mo--',linewidth=2, markersize=12)
plt.show()

plt.title("Confidence scores per layer - 5 images 8 layers")
plt.ylabel("Max confidence score")
plt.xlabel("DNN Layers")
#plt.style.use(['dark_background', 'presentation'])
plt.plot(layers, total_confidence_score[0], 'bo--',linewidth=2, markersize=12)
plt.plot(layers, total_confidence_score[1], 'go--',linewidth=2, markersize=12)
plt.plot(layers, total_confidence_score[2], 'ro--',linewidth=2, markersize=12)
plt.plot(layers, total_confidence_score[3], 'co--',linewidth=2, markersize=12)
plt.plot(layers, total_confidence_score[4], 'mo--',linewidth=2, markersize=12)
plt.show()


plt.title("Generated Labels matching the orginal prediction")
plt.ylabel("Count of accurate labels")
plt.xlabel("DNN Layers")
#plt.style.use(['dark_background', 'presentation'])
plt.plot(layers, total_confidence_labels[0], 'bo--',linewidth=2, markersize=12)
plt.plot(layers, total_confidence_labels[1], 'go--',linewidth=2, markersize=12)
plt.plot(layers, total_confidence_labels[2], 'ro--',linewidth=2, markersize=12)
plt.plot(layers, total_confidence_labels[3], 'co--',linewidth=2, markersize=12)
plt.plot(layers, total_confidence_labels[4], 'mo--',linewidth=2, markersize=12)
plt.show()
"""
