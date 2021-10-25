import numpy as np

def nms_cpu(dets, iou_thresh:float):
    """
    (0, 0) is at top left corner
    dets: numpy array
        [x_min, y_min, x_max, y_max, score]
    iou_thresh: float
        remove the box that has a lower confident if it's iou with a higher box is bigger than thresh hold
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    score = dets[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]
    keep = []
    while order.size > 1:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1) # set no intersection to 0
        intersection = w * h
        ious = (intersection) / (area[i] + area[order[1:]] - intersection)
        index = np.where(ious <= iou_thresh)[0]
        order = order[index + 1] # area[order[1:]] is shifted to left by 1
    return keep

def main():
    # test boxes
    dets = np.array([[100,120,170,200,0.98],
                    [20,40,80,90,0.99],
                    [20,38,82,88,0.96],
                    [200,380,282,488,0.9],
                    [19,38,75,91, 0.8]])
    keep = nms_cpu(dets, 0.7)
    print(keep)

if __name__ == '__main__':
    main()
