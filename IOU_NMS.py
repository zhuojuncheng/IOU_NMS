import numpy as np

predict_dict = {'dog': [
                    [0.88, 10, 5, 20, 25],
                    [0.68, 8, 8, 18, 18],
                    [0.45, 12, 10, 22, 20],

                    [0.98, 30, 10, 40, 20],
                    [0.48, 32, 8, 42, 18],
                    [0.78, 27, 12, 37, 22],
                    [0.50, 30, 13, 40, 23],
                    ],
                'cat': [
                    [0.65, 12, 10, 32, 30],
                    [0.32, 15, 8, 35, 28],
                    [0.85, 10, 9, 30, 29],
                    [0.63, 11, 5, 41, 35],
                ]}


def NMS(predicts_dict, threshold=0.3):
    for object_name, bounding_box in predicts_dict.items():  # 遍历字典中的置信度和框图坐标
        bounding_box_array = np.array(bounding_box, dtype=np.float)
        confidence = bounding_box_array[:, 0]
        x1 = bounding_box_array[:, 1]
        y1 = bounding_box_array[:, 2]
        x2 = bounding_box_array[:, 3]
        y2 = bounding_box_array[:, 4]
        areas = (x2 - x1 + 0.01) * (y2 - y1 + 0.01)
        order = confidence.argsort()[::-1]
        keep = []  # 用来存放去重叠后的框图坐标的索引信息
        while order.size > 0:
            i = order[0]
            keep.append(i)  # 保留当前最大置信度对应框图的索引
            x_1 = np.maximum(x1[i], x1[order[1:]])
            y_1 = np.maximum(y1[i], y1[order[1:]])
            x_2 = np.minimum(x2[i], x2[order[1:]])
            y_2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, x_2 - x_1) * np.maximum(0.0, y_2 - y_1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            # print("iou =", iou)
            indexs = np.where(iou <= threshold)[0] + 1
            order = order[indexs]
        bounding_box = bounding_box_array[keep]
        predicts_dict[object_name] = bounding_box.tolist()
    return predicts_dict


NMS(predict_dict)
print(predict_dict)
