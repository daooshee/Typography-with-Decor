# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torchvision.utils as vutils
from numpy import arccos, arcsin
from sklearn.cluster import DBSCAN

import numpy as np
import glob
import math

import copy
import warnings
import cv2
import queue as Queue
import os

import random

parser = argparse.ArgumentParser()
parser.add_argument('--style_path', help='path to the input style image')
parser.add_argument('--glyph_path', help='path to the corresponding glyph of the input style image')
parser.add_argument('--content_path', help='path to the target content image')
parser.add_argument('--transfered_path', help='path to the basal text effect transfer result')
parser.add_argument('--save_path', help='path to save the final result')
parser.add_argument('--style_name', help='name of the style image')
parser.add_argument('--content_name', help='name of the content image')

opt = parser.parse_args()

def get_area(data):
    return data['area']

class GetObjectTool():
    def get_search(self, i,j):
        self.record[i,j] = 1
        Object = np.zeros([256,256])
        Object[i,j] = 1

        x_min = self.img.shape[0]
        x_max = 0
        y_min = self.img.shape[1]
        y_max = 0

        q = Queue.Queue()
        q.put([i,j])

        while(not q.empty()):
            i,j = q.get()

            Object[i,j] = self.Map[i,j]
            x_min = min(i,x_min)
            y_min = min(j,y_min)
            x_max = max(i,x_max)
            y_max = max(j,y_max)

            for di,dj in [[0,-1],[-1,0],[0,1],[1,0]]:
                i_ = i+di
                j_ = j+dj
                if i_>0 and j_>0 and i_<self.img.shape[0] and j_<self.img.shape[1]:
                    if self.record[i_,j_] == 0 and self.Map[i_,j_] > 0:
                        self.record[i_,j_] = 1
                        q.put([i_,j_])

        return {'x_min':x_min,
                'y_min':y_min,
                'x_max':x_max+1,
                'y_max':y_max+1,
                'x_len':x_max-x_min+1,
                'y_len':y_max-y_min+1,
                'area':(x_max-x_min+1)*(y_max-y_min+1),
                'x_center':(x_min+x_max)/2,
                'y_center':(y_min+y_max)/2,
                'Object':self.img[x_min:x_max+1,y_min:y_max+1,:3],
                'Mask':Object[x_min:x_max+1,y_min:y_max+1]}

    def get(self, img, output_probs):
        self.img = img
        self.record = np.zeros([256,256])
        self.Map = output_probs[:,:,0]

        Obejcts = []
        for i in range(256):
            for j in range(256):
                if self.record[i,j] == 0 and self.Map[i,j] > 0:
                    Obejcts.append(self.get_search(i,j))

        NorSize = 5
        NormalizedObjects = np.zeros([len(Obejcts),NorSize*NorSize])

        for i in range(len(Obejcts)):
            NormalizedObjects[i,:] = cv2.resize(Obejcts[i]['Mask'],(NorSize,NorSize)).reshape((NorSize*NorSize))

        clustering = DBSCAN(eps=300, min_samples=1)
        clustering.fit(NormalizedObjects)
        labels = clustering.labels_

        labels_unique = np.unique(labels)

        ClassifiedObjects = []
        for i in range(len(labels_unique)):
            ClassifiedObjects.append([])
        for i in range(len(Obejcts)):
            ClassifiedObjects[labels[i]].append(Obejcts[i])
        for i in range(len(labels_unique)):
            ClassifiedObjects[i].sort(key=get_area, reverse=True)
        return ClassifiedObjects

    def computeMap(self, img):
        #### 1. Parameters
        scale = 1
        smaller_scale = 1
        kernel_scale = 0.9
        kernel_width_scale = 0.06

        #### 2. Image Map
        im_map = np.ones([256,256,3])

        # The distance map
        im_map[:,:,2] = ((1-img[:,:,0]/255.) ** 5) * 255

        # The convex hull
        img_gray = img[:,:,2]
        ret, thresh = cv2.threshold(img_gray, 127, 255,0)
        im, contours, hierarchy = cv2.findContours(thresh,2,1)
        img = img.copy()
        cv2.drawContours(img, contours,0,(0,0,255),1)

        for cnt in contours:
            hull = cv2.convexHull(cnt,returnPoints = False)

            defects = cv2.convexityDefects(cnt,hull)
            if(defects is None):
                continue

            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(img,start,end,[0,0,255],1)

        # The row map
        for j in range(img.shape[1]):
            i_start = -1
            i_end = -1

            for i in range(img.shape[0]):
                if img[i,j,2] > 50:
                    i_start = i
                    break
            for i in range(img.shape[0]-1,0,-1):
                if img[i,j,2] > 50:
                    i_end = i
                    break

            if (i_start != -1 and i_end != -1):
                kernel_width = int((i_end-i_start) * kernel_width_scale)
                for i in range(i_start-kernel_width,i_start+kernel_width+1):
                    im_map[i,j,0] += (kernel_width - np.abs(i-i_start)) * kernel_scale

                for i in range(i_end-kernel_width,i_end+kernel_width+1):
                    im_map[i,j,0] += (kernel_width - np.abs(i-i_end)) * kernel_scale

            for i in range(1,img.shape[0]):
                im_map[i,j,0] += im_map[i-1,j,0]
            im_map[:,j,0] -= im_map[img.shape[0]//2,j,0]

        # The col map
        for i in range(img.shape[0]):
            j_start = -1
            j_end = -1

            for j in range(img.shape[1]-1,0,-1):
                if img[i,j,2] > 50:
                    j_end = j
                    break
            for j in range(img.shape[1]):
                if img[i,j,2] > 50:
                    j_start = j
                    break
            j_centre = (j_end+j_start)//2

            if (j_start != -1 and j_end != -1):
                kernel_width = int((j_end-j_start) * kernel_width_scale)
                for j in range(j_start-kernel_width,j_start+kernel_width+1):
                    im_map[i,j,1] += (kernel_width - np.abs(j-j_start)) * kernel_scale

                for j in range(j_end-kernel_width,j_end+kernel_width+1):
                    im_map[i,j,1] += (kernel_width - np.abs(j-j_end)) * kernel_scale

            for j in range(1,img.shape[1]):
                im_map[i,j,1] += im_map[i,j-1,1]
            im_map[i,:,1] -= im_map[i,img.shape[1]//2,1]

        # normalize
        im_map[:,:,:2] -= im_map[:,:,:2].min()
        im_map[:,:,:2] /= im_map[:,:,:2].max()
        im_map[:,:,:2] *= 255
        im_map[:,:,:2] = cv2.GaussianBlur(im_map[:,:,:2],(25,25),5)

        return im_map

    def computeThicknessMap(self, img, size=3):
        Map = cv2.GaussianBlur(img[:,:,2],(25,25),15)
        return Map

    def computeDirMap(self, img, size=3):
        img = cv2.copyMakeBorder(img,size,size,size,size, cv2.BORDER_CONSTANT,value=[0,0,0])
        Ix = np.zeros(img.shape[:2]) + 1e-13
        Iy = np.zeros(img.shape[:2]) + 1e-13
        img[:,:,0] = img[:,:,2]
        img[:,:,1] = img[:,:,2]
        ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        img_new = img.copy()
        img_ori = img.copy()

        fil_y = np.zeros([size,size])
        fil_y[0,size//2] = -1
        fil_y[size-1,size//2] = 1

        fil_x = fil_y.T

        for i in range(img.shape[0]//2):
            binary = img[:,:,0]-0 # have to -0, else there will be TypeError 
            contours = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_new, contours[1],0,(255,255,255),2)
            delta = ((img_new-img) / 255.)[:,:,0]
            img_tmp = img_new[:,:,0]/255.
            Ix += cv2.filter2D(img_tmp,-1,fil_x) * delta
            Iy += cv2.filter2D(img_tmp,-1,fil_y) * delta
            img = img_new.copy()

        img = img_ori
        img_new = img.copy()
        for i in range(18):
            binary = img[:,:,0]-0 # have to -0, else there will be TypeError 
            contours = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_new, contours[1],0,(0,0,0),1)
            delta = ((img-img_new) / 255.)[:,:,0]
            img_tmp = img_new[:,:,0]/255.
            Ix += cv2.filter2D(img_tmp,-1,fil_x) * delta
            Iy += cv2.filter2D(img_tmp,-1,fil_y) * delta
            img = img_new.copy()

        dirmap = (np.arctan(Ix / Iy) / math.radians(180)) % 2
        dirmap_1 = Iy <= 0
        dirmap[dirmap_1 == True] += 1
        dirmap = dirmap % 2

        dirmap = cv2.filter2D(dirmap,-1,np.ones([5,5])/25) * (1-img_ori[:,:,0] / 255.)
        dirmap = dirmap[size:-size,size:-size]
        dirmap = cv2.GaussianBlur(dirmap,(25,25),3)
        return dirmap

    def search_nearest_L1(self, target_value, map):
        min_dis = 10000000.
        pos = [0,0]
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                value = map[i,j,:]
                dis = np.abs(target_value-value).mean()
                if min_dis > dis:
                    min_dis = dis
                    pos = [i,j]
        return pos

    def search_nearest_L2(self, target_value, map):
        min_dis = 10000000.
        pos = [0,0]
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                value = map[i,j,:]
                dis = ((target_value-value)**2).mean()
                if min_dis > dis:
                    min_dis = dis
                    pos = [i,j]
        return pos

    def search_nearest_L2_mask(self, target_value, map, mask):
        min_dis = 1e12
        pos = [-1,-1]

        i_len = mask.shape[0]
        j_len = mask.shape[1]

        for i in range(map.shape[0]-i_len):
            for j in range(map.shape[1]-j_len):
                value = map[i:i+i_len,j:j+j_len,:]
                dis = 0
                for c in range(map.shape[2]):
                    dis += (((target_value[:,:,c]-value[:,:,c])*mask)**2).mean()
                if min_dis > dis:
                    min_dis = dis
                    pos = [i+i_len//2,j+j_len//2]
        return pos, min_dis

    def rotate(self, image, angle):
        (h, w) = image.shape[:2]
        angle = angle * math.radians(180)
        new_w = int(round(abs(w * math.cos(angle)) + abs(h * math.sin(angle))))
        new_h = int(round(abs(h * math.cos(angle)) + abs(w * math.sin(angle))))
        if new_w % 2 == 0:
            new_w += 1
        if new_h % 2 == 0:
            new_h += 1
        delta_w = new_w - w
        delta_h = new_h - h
        if new_w > w :
            image = cv2.copyMakeBorder(image,0,0,delta_w//2,delta_w-delta_w//2, cv2.BORDER_CONSTANT,value=[0,0,0])
        if new_h > h :
            image = cv2.copyMakeBorder(image,delta_h//2,delta_h-delta_h//2,0,0, cv2.BORDER_CONSTANT,value=[0,0,0])
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle / math.radians(180) * 180, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def GenerateMaps(self, input_cha, output_cha, log_input, log_output, cache = False):
        if log_input and os.path.exists('cache/%s_pos.npy'%(log_input)) and cache:
            input_map = np.load('cache/%s_pos.npy'%(log_input))
        else:
            input_map = self.computeMap(input_cha)

        if log_input and os.path.exists('cache/%s_dir.npy'%(log_input)) and cache:
            input_dir_map = np.load('cache/%s_dir.npy'%(log_input))
        else:
            input_dir_map = self.computeDirMap(input_cha)

        if log_output and os.path.exists('cache/%s_pos.npy'%(log_output)) and cache:
            output_map = np.load('cache/%s_pos.npy'%(log_output))
        else:
            output_map = self.computeMap(output_cha)

        if log_output and os.path.exists('cache/%s_dir.npy'%(log_output)) and cache:
            output_dir_map = np.load('cache/%s_dir.npy'%(log_output))
        else:
            output_dir_map = self.computeDirMap(output_cha)
        return input_map,input_dir_map,output_map,output_dir_map


    def SaveMaps(self, input_map, input_dir_map, output_map, output_dir_map, log_input, log_output):
        np.save('cache/%s_pos.npy'%(log_input), input_map)
        np.save('cache/%s_dir.npy'%(log_input), input_dir_map)
        np.save('cache/%s_pos.npy'%(log_output), output_map)
        np.save('cache/%s_dir.npy'%(log_output), output_dir_map)


    def Move(self, input_img, input_cha, input_seg, output_img, output_cha, log_input, log_output):
        input_map, input_dir_map, output_map, output_dir_map = self.GenerateMaps(input_cha, output_cha, log_input, log_output)
        self.SaveMaps(input_map, input_dir_map, output_map, output_dir_map, log_input, log_output)

        input_thickness_map = self.computeThicknessMap(input_cha)
        output_thickness_map = self.computeThicknessMap(output_cha)

        input_map = np.concatenate((input_map,
                                    input_dir_map.reshape((input_img.shape[0], input_img.shape[1],1)) * 0,
                                    np.zeros([input_img.shape[0], input_img.shape[1], 1])), axis=2)
        output_map = np.concatenate((output_map,
                                     output_dir_map.reshape((input_img.shape[0], input_img.shape[1],1)) * 0,
                                    np.zeros([input_img.shape[0], input_img.shape[1], 1])), axis=2)

        input_Objects = self.get(input_img, input_seg)
        overlap_record = np.zeros(input_img.shape[:2])

        print('Find %d decor classes'%(len(input_Objects)))

        elements_num_ratio = (output_cha[:,:,2].mean()/input_cha[:,:,2].mean())

        loss_upper_bound = 15000
        for j in range(len(input_Objects)):
            decorationClass = input_Objects[j]
            print('Process the %d-th class, containing %d elements'%(j,len(decorationClass)), end='')

            #############################
            ##   Repeative Type
            #############################
            if len(decorationClass) > 2:
                # repeated decorative elements

                exchange_num = random.randint(1,len(decorationClass)-1)

                for k in range(len(decorationClass)):
                    cur_num = k % len(decorationClass)
                    next_num = (k+exchange_num) % len(decorationClass)

                    #### 1. Resize the element
                    mask = decorationClass[cur_num]['Mask']
                    new_size_ratio = (elements_num_ratio * (0.75+random.random()/2)) ** 0.2
                    new_size_x = int(mask.shape[0] * new_size_ratio)
                    new_size_y = int(mask.shape[1] * new_size_ratio)
                    mask = mask / 255.
                    mask = cv2.resize(mask,(new_size_x,new_size_y))
                    deco = decorationClass[cur_num]['Object']
                    deco = cv2.resize(deco,(new_size_x,new_size_y))

                    input_x_centre = decorationClass[next_num]['x_center'] + int(new_size_x/5 * random.random())
                    input_y_centre = decorationClass[next_num]['y_center'] + int(new_size_y/5 * random.random())

                    input_x_start = int(input_x_centre - mask.shape[0] // 2)
                    input_x_end = int(input_x_start + mask.shape[0])
                    input_y_start = int(input_y_centre - mask.shape[1] // 2)
                    input_y_end = int(input_y_start + mask.shape[1])

                    covering_shape_before = input_thickness_map[input_x_start:input_x_end,input_y_start:input_y_end] * mask

                    #### 2. Find the best place
                    output_cpos, min_dis = self.search_nearest_L2_mask(input_map[input_x_start:input_x_end,input_y_start:input_y_end,:], output_map, mask)

                    if output_cpos[0] < 0 or min_dis > loss_upper_bound:
                        continue

                    ### 3. Adjust
                    x_start = output_cpos[0] - mask.shape[0] // 2
                    x_end = x_start + mask.shape[0]
                    y_start = output_cpos[1] - mask.shape[1] // 2
                    y_end = y_start + mask.shape[1]

                    covering_shape_after = output_thickness_map[x_start:x_end,y_start:y_end] * mask

                    reshape_ratio = (covering_shape_after.mean()/covering_shape_before.mean()) ** 0.2
                    new_size_x = int(new_size_x * reshape_ratio)
                    new_size_y = int(new_size_y * reshape_ratio)

                    deco = cv2.resize(deco,(new_size_y,new_size_x))
                    mask = cv2.resize(mask,(new_size_y,new_size_x))

                    x_start = output_cpos[0] - mask.shape[0] // 2
                    x_end = x_start + mask.shape[0]
                    y_start = output_cpos[1] - mask.shape[1] // 2
                    y_end = y_start + mask.shape[1]

                    beyoung_boundary = False
                    while(x_start<5 or y_start<5 or x_end>output_img.shape[0]-5 or y_end>output_img.shape[1]-5):
                        if(new_size_x == int(new_size_x * 0.95) or new_size_y == int(new_size_y * 0.95)):
                            beyoung_boundary = True
                            break
                        new_size_x = int(new_size_x * 0.95)
                        new_size_y = int(new_size_y * 0.95)

                        deco = cv2.resize(deco,(new_size_y,new_size_x))
                        mask = cv2.resize(mask,(new_size_y,new_size_x))

                        x_start = output_cpos[0] - mask.shape[0] // 2
                        x_end = x_start + mask.shape[0]
                        y_start = output_cpos[1] - mask.shape[1] // 2
                        y_end = y_start + mask.shape[1]

                    if(beyoung_boundary):
                        continue

                    #### 4. Recompose
                    for c in range(3):
                        output_img[x_start:x_end,y_start:y_end,c] = output_img[x_start:x_end,y_start:y_end,c] * (1-mask) + deco[:,:,c] * mask

                    #### 5. Update Map
                    output_map[x_start:x_end,y_start:y_end,4] = output_map[x_start:x_end,y_start:y_end,4] * (1-mask) + 255 * mask

                    cv2.imwrite('temp/cur_recom_result.png', output_img)
                    print('.', end='')

            #############################
            ##   Single Type
            #############################
            else:
                for k in range(len(decorationClass)):
                    # single decorative elements
                    decoration = decorationClass[k]

                    #### 1. Resize the element
                    input_x_start = decoration['x_min']
                    input_x_end = decoration['x_max']
                    input_y_start = decoration['y_min']
                    input_y_end = decoration['y_max']

                    mask = decoration['Mask']
                    mask = mask / 255.
                    deco = decoration['Object']

                    covering_shape_before = input_thickness_map[input_x_start:input_x_end,input_y_start:input_y_end] * mask

                    gravity_input_map = np.zeros([decoration['x_len'],decoration['y_len'],2])
                    for i in range(decoration['x_len']):
                        for j in range(decoration['y_len']):
                            gravity_input_map[i,j,0] = i-(decoration['x_center']-input_x_start)
                            gravity_input_map[i,j,1] = j-(decoration['y_center']-input_y_start)
                    temp_map = input_cha[input_x_start:input_x_end,input_y_start:input_y_end,0]
                    gravity_input_map[:,:,0] *= temp_map * mask
                    gravity_input_map[:,:,1] *= temp_map * mask

                    #### 2. Find the best place
                    output_cpos,min_dis = self.search_nearest_L2_mask(input_map[input_x_start:input_x_end,input_y_start:input_y_end,:], output_map, mask)

                    if output_cpos[0] < 0 or min_dis > loss_upper_bound:
                        continue

                    #### 3. Adjust
                    x_start = output_cpos[0] - mask.shape[0] // 2
                    x_end = x_start + mask.shape[0]
                    y_start = output_cpos[1] - mask.shape[1] // 2
                    y_end = y_start + mask.shape[1]

                    covering_shape_after = output_thickness_map[x_start:x_end,y_start:y_end] * mask
                    reshape_ratio = (covering_shape_after.mean()/covering_shape_before.mean())
                    if reshape_ratio > 1:
                        reshape_ratio = reshape_ratio ** 0.5
                    reshape_ratio = reshape_ratio ** 0.5
                    new_size_x = int(decoration['x_len'] * reshape_ratio)
                    new_size_y = int(decoration['y_len'] * reshape_ratio)

                    if new_size_y <= 0 or new_size_x <= 0:
                        continue
                    deco = cv2.resize(deco,(new_size_y,new_size_x))
                    mask = cv2.resize(mask,(new_size_y,new_size_x))

                    x_start = output_cpos[0] - mask.shape[0] // 2
                    x_end = x_start + mask.shape[0]
                    y_start = output_cpos[1] - mask.shape[1] // 2
                    y_end = y_start + mask.shape[1]

                    beyoung_boundary = False
                    while(x_start<5 or y_start<5 or x_end>output_img.shape[0]-5 or y_end>output_img.shape[1]-5):
                        if(new_size_x == int(new_size_x * 0.95) or new_size_y == int(new_size_y * 0.95)):
                            beyoung_boundary = True
                            break
                        new_size_x = int(new_size_x * 0.95)
                        new_size_y = int(new_size_y * 0.95)

                        if new_size_x < 1 or new_size_y < 1:
                            beyoung_boundary = True
                            break

                        deco = cv2.resize(deco,(new_size_y,new_size_x))
                        mask = cv2.resize(mask,(new_size_y,new_size_x))

                        x_start = output_cpos[0] - mask.shape[0] // 2
                        x_end = x_start + mask.shape[0]
                        y_start = output_cpos[1] - mask.shape[1] // 2
                        y_end = y_start + mask.shape[1]

                    if(beyoung_boundary):
                        continue

                    # When resizing the element, we also need to shift it a little
                    gravity_map = np.zeros([x_end-x_start,y_end-y_start,2])
                    for i in range(x_end-x_start):
                        for j in range(y_end-y_start):
                            gravity_map[i,j,0] = i-(output_cpos[0]-x_start)
                            gravity_map[i,j,1] = j-(output_cpos[1]-y_start)
                    temp_map = input_cha[x_start:x_end,y_start:y_end,0]
                    gravity_map[:,:,0] *= temp_map * mask
                    gravity_map[:,:,1] *= temp_map * mask

                    output_cpos[0] += int((gravity_map[:,:,0].mean()-gravity_input_map[:,:,0].mean()) * 0.15)
                    output_cpos[1] += int((gravity_map[:,:,1].mean()-gravity_input_map[:,:,1].mean()) * 0.15)

                    x_start = output_cpos[0] - mask.shape[0] // 2
                    x_end = x_start + mask.shape[0]
                    y_start = output_cpos[1] - mask.shape[1] // 2
                    y_end = y_start + mask.shape[1]

                    #### 4. Recompose
                    for c in range(3):
                        output_img[x_start:x_end,y_start:y_end,c] = output_img[x_start:x_end,y_start:y_end,c] * (1-mask) + deco[:,:,c] * mask
                    #### 5. Update Map
                    output_map[x_start:x_end,y_start:y_end,4] = output_map[x_start:x_end,y_start:y_end,4] * (1-mask) + 255 * mask
                    cv2.imwrite('temp/cur_recom_result.png', output_img)
                    print('.', end='')

            print()
        print()
        return output_img, input_map, output_map, input_dir_map, output_dir_map


warnings.filterwarnings("ignore")

###### Parameters ######
ngf = 64
###### Parameters ######

GOT = GetObjectTool()

Img = cv2.imread(opt.style_path)
Img = cv2.resize(Img,(256,256))

Seg = cv2.imread('temp/mask_final.jpg')
Seg = cv2.resize(Seg,(256,256))

Cha = cv2.imread(opt.glyph_path)
Cha = cv2.resize(Cha,(256,256))

TargetCha = cv2.imread(opt.content_path)
TargetCha = cv2.resize(TargetCha,(256,256))

TargetImg = cv2.imread(opt.transfered_path)
TargetImg = cv2.resize(TargetImg,(256,256))

result, input_map, output_map, input_dir_map, output_dir_map = GOT.Move(Img,Cha,Seg,TargetImg,TargetCha, opt.style_name, opt.content_name)
cv2.imwrite(opt.save_path, result)
