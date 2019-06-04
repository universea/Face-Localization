import sys
import cv2
import six
import os
import math
import time
import numpy as np
from tqdm import tqdm

# 计算两个框的iou
def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)
# 将框整理为正方形
def get_square_box(box):
    """Get the square boxes which are ready for CNN from the boxes"""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return left_x, top_y, right_x, bottom_y

# 读取标注
def read_points(file_name=None):
    """
    Read points from .pts file.
    """
    points = []
    with open(file_name) as file:
        line_count = 0
        for line in file:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                loc_x, loc_y = line.strip().split(sep=" ")
                points.append([float(loc_x), float(loc_y)])
                line_count += 1
    return points

# 绘制关键点
def draw_landmark_point(image, points):
    """
    Draw landmark point on image.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 1, (0, 255, 0), -1, cv2.LINE_AA)

def make_datasets(image,gray,points):
    # 探测图片中的人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.5,
        minNeighbors = 5,
        minSize = (5,5)
    )
    #print ("发现{0}个人脸!".format(len(faces)))
    #print(faces)
    points_translate = np.array(points).T
    xmin = min(points_translate[0]).astype(np.int32)
    xmax = max(points_translate[0]).astype(np.int32)
    ymin = min(points_translate[1]).astype(np.int32)
    ymax = max(points_translate[1]).astype(np.int32)

    box = [0,0,0,0]
    minimal_box = [xmin,ymin,xmax,ymax]
    for(x,y,w,h) in faces:
        box[0] = x
        box[1] = y
        box[2] = x + w
        box[3] = y + h
        iou = compute_iou(np.array(box),np.array(minimal_box))
        #print(iou)
        if iou > 0.1:
            #cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(0,255,128),3)
            if minimal_box[0] > box[0]:
                minimal_box[0] = box[0]
            if minimal_box[1] > box[1]:
                minimal_box[1] = box[1]
            if minimal_box[2] < box[2]:
                minimal_box[2] = box[2]
            if minimal_box[3] < box[3]:
                minimal_box[3] = box[3]

    w = minimal_box[2] - minimal_box[0]
    h = minimal_box[3] - minimal_box[1]
    #print("w,h = ",w,h)

    xmin,ymin,xmax,ymax = get_square_box(minimal_box)

    points_translate[0] = points_translate[0] - xmin
    points_translate[1] = points_translate[1] - ymin

    cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(126,0,25),2)
    slip_image = image[ymin:ymax,xmin:xmax]

    #draw_landmark_point(image,points)
    

    wh = xmax - xmin
    slip_image   = cv2.resize(slip_image, (224,224), interpolation=cv2.INTER_CUBIC)
    slip_image_wh = slip_image.shape[0]
    xishu = slip_image_wh/wh
    #print(wh)
    #print(slip_image_wh)
    #print(xishu)
    points_translate = points_translate*xishu

    #draw_landmark_point(slip_image,points_translate.T)
    #cv2.imshow("Find Faces!",image)
    #cv2.imshow("slip_image!",slip_image)
    #cv2.waitKey(0)
    return slip_image,points_translate.T

image_names = []
for image_name in os.listdir("dataset/300w/01_Indoor"):
    if image_name.split('.')[-1] in ['png']:
        image_names.append("dataset/300w/01_Indoor/"+image_name)
for image_name in os.listdir("dataset/300w/02_Outdoor"):
    if image_name.split('.')[-1] in ['png']:
        image_names.append("dataset/300w/02_Outdoor/"+image_name)
print(image_names)

# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

for image_name in image_names:
    # 待检测的图片路径
    points_name = image_name.replace('.png', '.pts')
    points    = read_points(points_name)

    # 读取图片
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    result_image,result_points = make_datasets(image,gray,points)
    result_image_name = image_name.split('door/')[1]
    print(result_image_name)
    cv2.imwrite("dataset/300w_224x224/"+result_image_name,result_image)

    result_points_name = result_image_name.replace('.png', '.pts')

    with open("dataset/300w_224x224/"+result_points_name, 'w+') as f:
        f.write("version: 1"+'\n')   #加\n换行显示
        f.write("n_points: 68"+'\n')   #加\n换行显示
        f.write("{"+'\n')   #加\n换行显示
        for line in result_points:
            f.write(str(line[0])+" "+str(line[1])+'\n')
        f.write("}"+'\n')   #加\n换行显示
