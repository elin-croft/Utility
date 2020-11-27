###################################################
#                    notice
#                ratio is w / h
###################################################

import os, sys
import numpy as np
from xml.etree.ElementTree import ElementTree


def get_iou(bboxes, cluster):
    """
    param:
    bboxes: numpy array with shape of (n, 2)
    n boxes in you dataset
    cluster: numpy array with shape of (k, 2)
             k centers
    """
    # we assume that all bbox starts at point (0, 0), so we only need width and height of those boxes
    w_min = np.minimum(bboxes[:, 0, np.newaxis], cluster[np.newaxis, :, 0]) # shape (n, k)
    h_min = np.minimum(bboxes[:, 1, np.newaxis], cluster[np.newaxis, :, 1]) # shape (n, k)

    intersection = w_min * h_min # shape (n, k)

    # calculate union area A or B - A and B 
    # more details please check venn diagram
    boxes_area = bboxes[:, 0] * bboxes[:, 1] # shape (n, )
    cluster_area = cluster[:, 0] * cluster[:, 1] # shape (, k)
    union = (boxes_area[:, np.newaxis] + cluster_area[np.newaxis, :]) - intersection
    iou = intersection / union # shape (n, k)
    
    return iou

class Kmeans:
    def __init__(self, root, k):
        self.root = root
        self.k = k
        self.dataset_path = os.path.join(self.root, 'Annotations')
        self.boxes = self.get_boxes()

    def get_boxes(self):
        box = []
        et = ElementTree()
        files = os.listdir(self.dataset_path)
        
        for xml_file in files:
            if xml_file.endswith('.xml'):
                et.parse(os.path.join(self.dataset_path, xml_file))
                root = et.getroot()
                for size in root.iter('size'):
                    w_pic = float(size.find('width').text)
                    h_pic = float(size.find('height').text)
                size = (float())
                for bboxes in root.iter('bndbox'):
                    xmin = float(bboxes.find('xmin').text)
                    xmax = float(bboxes.find('xmax').text)
                    ymin = float(bboxes.find('ymin').text)
                    ymax = float(bboxes.find('ymax').text)
                    shape = [round((xmax - xmin), 2), round((ymax - ymin), 2)]
                    box.append(shape)
        return np.array(box)

    def init_centroid(self, boxes, k)->np.ndarray:
        centroids = None
        n = boxes.shape[0] - 1
        
        first_centroid = np.random.randint(0, n, 1)
        centroids = boxes[first_centroid, :]
        for index in range(k-1):
            ious = get_iou(boxes, centroids)
            loss = 1 - ious
            # distance between centroid and itself is 0 for there iou is 1
            labels = np.argmin(loss,axis=1)
            distance_sum = np.sum(loss[np.arange(len(labels)), labels])
            m = 0
            r = np.random.random() * distance_sum
            for i in range(n):
                if r <= m:
                    centroids = np.append(centroids, boxes[np.newaxis, i], axis=0)
                    break
                m += loss[i, labels[i]]
        return centroids
    
    def start(self,init='kmeans++'):
        boxes = self.get_boxes()
        n = len(boxes)
        last_iou = None
        last_label = None
        if init == 'random':
            anchors = boxes[np.random.choice(n, self.k)]
        elif init == 'kmeans++':
            anchors = self.init_centroid(boxes, self.k)
        while True:
            ious = get_iou(boxes, anchors)
            last_iou = ious
            loss = 1 - last_iou
            labels = np.argmin(loss, axis=1)
            if (last_label == labels).all():
                break
            for i in range(self.k):
                anchors[i] = np.mean(boxes[labels == i], axis=0)
            last_label = labels
        avg_iou = self.avg_iou(last_iou, last_label)
        print(avg_iou)
        print(anchors[:, 0] / anchors[:, 1])
        print(anchors)

    def avg_iou(self, ious, labels):
        return np.mean(ious[np.arange(len(labels)), labels])
 

k_mean = Kmeans('/mnt/d/test/model_data/model/weixing_OD_Train/budai_OD/data/dataset', 9)
k_mean.start()