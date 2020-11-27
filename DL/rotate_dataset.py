#coding=utf-8

import cv2
import math
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
import re

#旋转图像的函数
def rotate_image(src, angle, scale=1.):
    h,w,_ = np.shape(src)

    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # 仿射变换
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def rotate_xml(src, xmin, ymin, xmax, ymax, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # 获取旋转后图像的长和宽
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # rot_mat是最终的旋转矩阵
    # 获取原始矩形的四个中点，然后将这四个点转换到旋转后的坐标系下
    point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
    point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
    point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
    point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
    # 合并np.array
    concat = np.vstack((point1, point2, point3, point4))
    # 改变array类型
    concat = concat.astype(np.int32)

    c = np.array([[concat[0]],[concat[1]],[concat[2]],[concat[3]]])
    rx, ry, rw, rh = cv2.boundingRect(c)
    return rx, ry, rw, rh

if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    angle = sys.argv[3]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    jpg_in_dir = os.path.join(in_dir, 'JPEGImages')
    xml_in_dir = os.path.join(in_dir, 'Annotations')
    jpg_out_dir = os.path.join(out_dir, 'JPEGImages')
    if not os.path.exists(jpg_out_dir):
        os.makedirs(jpg_out_dir)
    xml_out_dir = os.path.join(out_dir, 'Annotations')
    if not os.path.exists(xml_out_dir):
        os.makedirs(xml_out_dir)

    images = os.listdir(jpg_in_dir)
    # num = imgs_num = len([lists for lists in os.listdir(jpg_in_dir)])
    num = len(images)
    i =0
    for img in images:
        i += 1
        name, ext = os.path.splitext(img)
        #print name
        # Images
        jpg_in_file = os.path.join(jpg_in_dir, name+".jpg")
        #print jpg_in_file
        img = cv2.imread(jpg_in_file)
        rotated_img = rotate_image(img, float(angle))
        
        jpg_out_file = os.path.join(jpg_out_dir, '{}_{}_{}.jpg'.format(name, "rotate", angle))
        jpg_in_name = '{}{}'.format(name, ".jpg")
        jpg_out_name = os.path.basename(jpg_out_file)
        cv2.imwrite(jpg_out_file, rotated_img)

        # Annotations
        xml_in_file = os.path.join(xml_in_dir, name+".xml")
        tree = ET.parse(xml_in_file)
        root = tree.getroot()

        for box in root.iter('bndbox'):
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            x, y, w, h = rotate_xml(img, xmin, ymin, xmax, ymax, float(angle))
            # 改变xml中的人脸坐标值
            box.find('xmin').text = str(x)
            box.find('ymin').text = str(y)
            box.find('xmax').text = str(x+w)
            box.find('ymax').text = str(y+h)
            box.set('updated', 'yes')

        xml_out_file = os.path.join(xml_out_dir, '{}_{}_{}.xml'.format(name, "rotate", angle))
        tree.write(xml_out_file)
        # 改变xml中的picture name
        xml = open(xml_out_file, 'r')
        lines = xml.readlines()
        xml.close()
        xml = open(xml_out_file, 'w+')
        print(jpg_in_name, jpg_out_name)
        for s in lines:
            a = re.sub(jpg_in_name,str(jpg_out_name), s)
            xml.writelines(a)
        xml.close()

        print('{}/{} {}'.format(i, num, name))
