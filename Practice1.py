# @FileName：Practice1.py
# @Description：
# @Author：dyh
# @Time：2023/4/5 14:57
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np
from imutils import contours

import myutils


def cv_show(im):
    cv2.imshow('demo', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
    第一大步先处理模板
'''
# 读取模板图
template = cv2.imread('img3/ocr_a_reference.png')
cv_show(template)
# 将模板图转为灰度图
temp_gary = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cv_show(temp_gary)
# 二值图像 这里大于10像素的都取0即黑色，不大于10的都取255即白色
_, temp_thresh = cv2.threshold(temp_gary, 10, 255, cv2.THRESH_BINARY_INV)
cv_show(temp_thresh)
# 计算轮廓
# cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）
# cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
# 返回的list中每个元素都是图像中的一个轮廓
binary, temp_contours, _ = cv2.findContours(temp_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 画出每个轮廓
temp_draw = cv2.drawContours(template.copy(), temp_contours, -1, (0, 0, 255), 2)
cv_show(temp_draw)

# 排序，从左到右，从上到下
temp_contours = myutils.sort_contours(temp_contours, method="left-to-right")[0]

temp_res_dict = {}

# 遍历每一个轮廓
for (i, c) in enumerate(temp_contours):
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    single_temp_num = temp_thresh[y:y + h, x:x + w]
    single_temp_num = cv2.resize(single_temp_num, (57, 88))
    temp_res_dict[i] = single_temp_num

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

'''
    第二大步处理信用卡
'''
# 读取信用卡图
img = cv2.imread('img3/credit_card_03.png')    
cv_show(img)
img = myutils.resize(img, width=300)
# 转为灰度图
img_gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show(img_gary)

# 礼帽操作，突出更明亮的区域
mor_top_img = cv2.morphologyEx(img_gary, cv2.MORPH_TOPHAT, rectKernel)
cv_show(mor_top_img)
# Sobel 算子
xSobel = cv2.Sobel(mor_top_img, cv2.CV_64F, 1, 0, ksize=-1)  # ksize=-1相当于用3*3的
xSobel = np.absolute(xSobel)
# 取最大最小值做归一化
minVal = np.min(xSobel)
maxVal = np.max(xSobel)
xSobel = (255 * ((xSobel - minVal) / (maxVal - minVal)))
xSobel = xSobel.astype("uint8")
cv_show(xSobel)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
xSobel = cv2.morphologyEx(xSobel, cv2.MORPH_CLOSE, rectKernel)
# 通过二值操作 将图片要获取的区域更凸显
# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
_, thresh2 = cv2.threshold(xSobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv_show(thresh2)

# 在做一个闭运算，将图片中的白色区域都链接起来，即将数字一组一组的标记出来
mor_close2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, sqKernel)
cv_show(mor_close2)

# 计算轮廓
binary, mor_contours, _ = cv2.findContours(mor_close2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
draw_con_img = cv2.drawContours(img.copy(), mor_contours, -1, (0, 0, 255), 2)
cv_show(draw_con_img)
# 遍历轮廓，根据宽高过滤出四组数字的轮廓
locs = []
for (i, c) in enumerate(mor_contours):
    # 计算矩形
    x, y, w, h = cv2.boundingRect(c)
    aor = w / float(h)
    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if 2.5 < aor < 4.5:
        # 符合的留下来
        if (40 < w < 55) and (10 < h < 20):
            locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda j: j[0])

# 遍历每组轮廓的单个数字和模板进行匹配
output = []
for (i, (gx, gy, gw, gh)) in enumerate(locs):
    group_output = []
    # 像素都加5，往外边缘取一点
    group_num = img_gary[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]
    cv_show(group_num)

    # 对组进行二值化
    _, group_num_thresh = cv2.threshold(group_num, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 计算每一组的轮廓
    _, group_num_contours, _ = cv2.findContours(group_num_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 对计算的轮廓进行从左到右排序
    group_num_contours = contours.sort_contours(group_num_contours, method='left-to-right')[0]
    # 遍历计算每一组中的每一个数值
    for (i, c) in enumerate(group_num_contours):
        # 找到当前数值的轮廓，resize成合适的的大小
        x, y, w, h = cv2.boundingRect(c)
        single_num = group_num_thresh[y:y + h, x:x + w]
        single_num = cv2.resize(single_num, (57, 88))
        cv_show(single_num)

        # 拿到数值和模板匹配，将匹配上的索引记录下来即是该数字
        # 计算匹配得分
        scores = []
        for (num, t) in temp_res_dict.items():
            match_res = cv2.matchTemplate(single_num, t, cv2.TM_CCORR)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(match_res)
            scores.append(maxVal)

        # 找出最的大分数
        group_output.append(str(np.argmax(scores)))

    # 将这一组匹配的结果画出来
    group_temp_res = cv2.rectangle(img.copy(), (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 2)
    cv_show(group_temp_res)
    cv2.putText(img, "".join(group_output), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # 得到结果
    output.extend(group_output)

cv_show(img)
