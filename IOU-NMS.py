import numpy as np


def iou(box, boxes, isMin=False):
    # 计算全面积
    area = (box[2] - box[0]) * (box[3] - box[1])
    areaes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 提取重叠部分坐标
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    # 计算相交面积
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    if isMin:
        iou = np.divide(inter, np.minimum(area, areaes))
    else:
        iou = np.divide(inter, (area + areaes - inter))
    return iou


def nms(boxes, threshold=0.3, isMin=False):
    # 置信度从大到小排序
    _boxes = boxes[np.argsort(-boxes[:, 4])]
    #放置保留下来的方框信息
    keep = []
    while _boxes.size > 0:
        box = _boxes[0]
        b_boxes = _boxes[1:]
        keep.append(box)
        index = np.where(iou(box, b_boxes, isMin) < threshold)
        _boxes = b_boxes[index]
    return np.stack(keep)


if __name__ == '__main__':
    a = np.array(
        [
            [10, 5, 20, 25, 0.88],
            [8, 8, 18, 18, 0.68],
            [12, 10, 22, 20, 0.45],

            [30, 10, 40, 20, 0.98],
            [32, 8, 42, 18, 0.48],
            [27, 12, 37, 22, 0.78],
            [30, 13, 40, 23, 0.50],
        ]
    )
    c = np.array([[3, 3, 60, 60, 0.72], [4, 4, 40, 40, 0.63], [56, 56, 87, 87, 0.68]])
    print(nms(c))
