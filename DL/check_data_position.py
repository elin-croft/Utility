import os, sys
import cv2

import tqdm
import numpy as np

try:
    from xml.etree.cElementTree import ElementTree
except ModuleNotFoundError as e:
    from xml.etree.ElementTree import ElementTree

def draw_boxes(dataset_path, fix="", dst=None, flavour=None):
    folderName = os.path.split(dataset_path)[-1]
    savePath = os.path.join(dataset_path, 'show_bbox') if dst is None else os.path.join(dst, folderName, 'show_bbox')
    et = ElementTree()
    annoPath = None
    imgPath = None
    if flavour == 'std_voc':
        annoPath = 'data/VOCdevkit/VOC2007/Annotations'
        imgPath = 'data/VOCdevkit/VOC2007/JPEGImages'
    files = os.listdir(os.path.join(dataset_path, fix, annoPath))
    print("ready to process pictures")
    for xml_file in tqdm.tqdm(files):
        if xml_file.endswith('.xml'):
            img = cv2.imread(os.path.join(dataset_path, fix, imgPath, xml_file[:-4] + '.jpg'))
            et.parse(os.path.join(dataset_path, fix, annoPath, xml_file))
            root = et.getroot()
            for item in root.iter('object'):
                tyepId = item.find('name').text
                for bbox in item.iter('bndbox'):
                    xmin = int(bbox.find('xmin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymin = int(bbox.find('ymin').text)
                    ymax = int(bbox.find('ymax').text)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    if ymin > 50:
                        cv2.putText(img, tyepId, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(img, tyepId, (xmin, ymax + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                cv2.imwrite(os.path.join(savePath, xml_file[:-4] + '_checked.jpg'), img)
    print("done")


if __name__ == "__main__":
    root = sys.argv[1]
    dst = None
    if len(sys.argv) > 2:
        dst = sys.argv[2]
    fix = "mmdetection"
    root = root[:-1] if root[-1] == '/' else root
    draw_boxes(root, fix=fix, dst=dst, flavour='std_voc')
