# @FileName：Demo1.py
# @Description：
# @Author：dyh
# @Time：2023/3/27 20:34
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np


def cv_show(im):
    cv2.imshow('demo', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
    读取图像
'''
# im = cv2.imread('img1/cat.jpg',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('img', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
'''
    读取视频
'''
# vc = cv2.VideoCapture('img1/test.mp4')
# if vc.isOpened():
#     flag, frame = vc.read()
# else:
#     flag = False
#
# while flag:
#     ret, frame = vc.read()
#     if frame is None:
#         break
#     if ret:
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('result', img)
#         if cv2.waitKey(10) & 0xFF == 27:
#             break
# vc.release()
# cv2.destroyAllWindows()
'''
    截取图像
'''
# img = cv2.imread('img1/cat.jpg')
# crop_img = img[0:100, 0:200]
# cv2.imshow('crop', crop_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
    分别保留图像的BGR
'''
# img = cv2.imread('img1/cat.jpg')
# b, g, r = cv2.split(img)
# print(b.shape, g.shape, r.shape)
# img2 = cv2.merge((b, g, r))
# print(img2.shape)
# 只保留 b
# img[:, :, 1] = 0
# img[:, :, 2] = 0
# cv_show(img)

# 只保留 g
# img[:, :, 0] = 0
# img[:, :, 2] = 0
# cv_show(img)

# 只保留 r
# img[:, :, 0] = 0
# img[:, :, 1] = 0
# cv_show(img)

'''
    边缘填充
'''
# img = cv2.imread('img1/cat.jpg')
# top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
# replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
# cv_show(replicate)
# reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
# cv_show(reflect)
# reflect_101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101)
# cv_show(reflect_101)
# wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
# cv_show(wrap)
# constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,value=0)
# cv_show(constant)
'''
    图像数值操作
        直接使用图片 + 数值：这种情况相加超过255的对256进行取余，如相加结果为257，则该像素位为1
        使用cv2.add(img1,img2)：这总情况超过255的算255，如相加结果为257，则该像素位为255
        两个图片相加的前提是宽高相同
'''
img_cat = cv2.imread('img1/cat.jpg')
# img_dog = cv2.imread('img1/dog.jpg')
# # 返回值是高，宽，和几个通道
# h, w, _ = img_cat.shape
# print(h, w)
# # resize 是宽，高
# img_dog = cv2.resize(img_dog, (w, h))
# # 参数分别为：图片1， 图片1的权重， 图片2， 图片2的权重，提亮数值
# res = cv2.addWeighted(img_cat, 0.5, img_dog, 0.5, 0)
# cv_show(res)
'''
    知识补充：
        在原来的图片宽高上，对宽和高扩大四倍，fx比哦是对宽扩大的倍数，fy表示对高的扩大倍数
'''
# res = cv2.resize(img_cat, (0, 0), fx=2, fy=1)
# cv_show(res)

'''
    形态学-图片腐蚀：选取一个指定区域，然后如果该区域存在两种颜色则将内层的颜色腐蚀为外层颜色
        kernel:核，表示创建多大的卷积区域，越大腐蚀效果越严重
        iterations：腐蚀迭代次数
'''
# img = cv2.imread('img1/dige.png')
# kernel = np.ones((3, 3), np.uint8)
# erosion = cv2.erode(img, kernel, iterations=2)
# cv_show(erosion)

'''
    形态学-图片膨胀：他和腐蚀正好相反，他是将选取区域中的外层颜色变为内层颜色
        kernel:核，表示创建多大的卷积区域，越大膨胀效果越严重
        iterations：膨胀迭代次数
'''
# img = cv2.imread('img1/dige.png')
# kernel = np.ones((3, 3), np.uint8)
# erosion = cv2.erode(img, kernel, iterations=2)
# # 先腐蚀，在膨胀
# dilate = cv2.dilate(erosion, kernel, iterations=1)
# cv_show(dilate)
'''
    开运算：先腐蚀在膨胀，原理同上面的腐蚀膨胀
'''
# img = cv2.imread('img1/dige.png')
# kernel = np.ones((3, 3), np.uint8)
# morpholog = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
# cv_show(morpholog)

'''
    闭运算：先膨胀在腐蚀，原理同上面的腐蚀膨胀
'''
# img = cv2.imread('img1/dige.png')
# kernel = np.ones((3, 3), np.uint8)
# morpholog = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
# cv_show(morpholog)

'''
    梯度运算：一张图的 膨胀结果 - 腐蚀结果，得到膨胀图中比腐蚀多出来的那一圈
'''
# img = cv2.imread('img1/pie.png')
# kernel = np.ones((10, 10), np.uint8)
# mor = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# cv_show(mor)

'''
    礼帽：原始输入-开运算结果，相当于只要图片中的边缘多余出来的部分
'''
# img = cv2.imread('img1/dige.png')
# kernel = np.ones((5, 5), np.uint8)
# mor = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# cv_show(mor)

'''
    黑帽：闭运算-原始输入，相当于只要图片中去除多余边缘后原图片的轮廓
'''
# img = cv2.imread('img1/dige.png')
# kernel = np.ones((5, 5), np.uint8)
# mor = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# cv_show(mor)

'''
    图像梯度-Sobel算子：梯度即图像边界
        他的算法思想是根据卷积核大小选取一个区域，然后在区域中选取一个点，在X水平轴上对该点
        进行 (右边 - 左边) 的计算，因为相减存在复数，而色值是0~255，所以要对相减的结果取绝对值。
        在Y垂直轴上进行相同的操作，点的 (下边 - 上边) 取绝对值，然后将X轴和Y轴结果相加即可算出
        图像边界
        
        Sobel函数：dst = cv2.Sobel(src, ddepth, dx, dy, ksize)
            ddepth：图像的深度，一般使用 -1
            dx：X水平轴，0表示不计算，1表示计算
            dy：Y垂直轴，0表示不计算，1表示计算
            ksize：Sobel算子的大小(即选取区域大小)
            
        参数ddepth:
            在函数cv2.Sobel()的语法中规定，可以将函数cv2.Sobel()内ddepth参数的值设置为-1，
            让处理结果与原始图像保持一致。但是，如果直接将参数ddepth的值设置为-1，在计算时得到
            的结果可能是错误的。
            在实际操作中，计算梯度值可能会出现负数。如果处理的图像是8位图类型，则在ddepth的参数
            值为-1时，意味着指定运算结果也是8位图类型，那么所有负数会自动截断为0，发生信息丢失。
            为了避免信息丢失，在计算时要先使用更高的数据类型cv2.CV_64F，再通过取绝对值将其映射为
            cv2.CV_8U（8位图）类型。通常要将函数cv2.Sobel()内参数ddepth的值设置为“cv2.CV_64F”。
            要将偏导数取绝对值，以保证偏导数总能正确地显示出来。
            <<说白了就是先将负数转化为cv2.CV_64F类型，等后面使用convertScaleAbs函数取绝对值
            的时候会将cv2.CV_64F转为原来的负数，不取绝对值的话会有丢失>>
      
        注意：在使用Sobel函数时dx, dy不建议都给1，而是使用 水平结果 + 垂直结果，
        这样取出来的图像梯度效果会更好   
'''
# img = cv2.imread('img1/pie.png')
# # X轴不取绝对值
# xSobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, 3)
# cv_show(xSobel)
# # X轴取绝对值
# absXSobel = cv2.convertScaleAbs(xSobel)
# cv_show(absXSobel)
# # Y轴不取绝对值
# ySobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, 3)
# cv_show(ySobel)
# # Y轴取绝对值
# absYSobel = cv2.convertScaleAbs(ySobel)
# cv_show(absYSobel)
# # X轴绝对值 + Y轴绝对值
# sobelxy = cv2.addWeighted(absXSobel, 0.5, absYSobel, 0.5, 0)
# cv_show(sobelxy)
#
# # 测试2
# img2 = cv2.imread('img1/lena.jpg', cv2.IMREAD_GRAYSCALE)
# sobelX = cv2.Sobel(img2, cv2.CV_64F, 1, 0, 3)
# sobelX = cv2.convertScaleAbs(sobelX)
# sobelY = cv2.Sobel(img2, cv2.CV_64F, 0, 1, 3)
# sobelY = cv2.convertScaleAbs(sobelY)
# sobelXY = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
# cv_show(sobelXY)

'''
    图像梯度-Scharr算子：
        他只是在Sobel算子上做的更加细致了，不需要设置ksize，取得内容更多了，算法思想一样
'''
# img = cv2.imread('img1/lena.jpg',cv2.IMREAD_GRAYSCALE)
# charrX = cv2.Scharr(img,cv2.CV_64F, 1, 0)
# charrX = cv2.convertScaleAbs(charrX)
# charrY = cv2.Scharr(img,cv2.CV_64F, 0, 1)
# charrY = cv2.convertScaleAbs(charrY)
# charrXY = cv2.addWeighted(charrX, 0.5, charrY, 0.5, 0)
# cv_show(charrXY)

'''
    图像梯度-laplacian算子：
        他的算法思想是取一个卷积核的中心点，然后比较中心点 上下左右相加的结果 和 中心点*4的结果，
        它不存在X轴和Y轴的情况，因为他对噪音点比较敏感，所以一般不会单独使用
'''
# img = cv2.imread('img1/lena.jpg', cv2.IMREAD_GRAYSCALE)
# res = cv2.Laplacian(img, cv2.CV_64F)
# res = cv2.convertScaleAbs(res)
# cv_show(res)

'''
    图像平滑处理-均值滤波：
        算法思想就是去一个指定大小的卷积核（注意卷积核取值一般为奇数），然后将卷积核中的每个像素点
        相加，将相加结果在除以卷积核中像素点的总个数算出平均值，为要改变的点设置这个平均值像素
'''
# img = cv2.imread('img1/lenaNoise.png')
# res = cv2.blur(img, (3, 3))
# cv_show(res)

'''
    图像平滑处理-方框滤波：
        和上面均值滤波实现方式一模一样，就是多了一个是否除以卷积核总像素点个数的选项normalize
        ddepth = -1 表示保持和原图片颜色一致
        normalize = true 结果和上面均值滤波一样，就是要求平均
        normalize = false 表示不求平均，那么相加的结果超过255，就会按255算
'''
# img = cv2.imread('img1/lenaNoise.png')
# res1 = cv2.boxFilter(img, -1, (3, 3), normalize=True)
# cv_show(res1)
# res2 = cv2.boxFilter(img, -1, (3, 3), normalize=False)
# cv_show(res2)

'''
    图像平滑处理-高斯滤波：
        高斯模糊的卷积核里的数值是满足高斯分布，
        相当于更重视中间的，卷积核内据里中间点越近的权重越高，越远的权重越低
        sigmaX：在X轴上的标准差
'''
# img = cv2.imread('img1/lenaNoise.png')
# res = cv2.GaussianBlur(img, (3, 3), 1)
# cv_show(res)

'''
    图像平滑处理-中值滤波：
        就是对指定大小的卷积核内的像素点进行排序，然后取排序后的中间值
'''
# img = cv2.imread('img1/lenaNoise.png')
# res = cv2.medianBlur(img, 3)
# cv_show(res)

'''
    图像阈值操作：ret, dst = cv2.threshold(src, thresh, maxval, type)
        src： 输入图，只能输入单通道图像，通常来说为灰度图
        dst： 输出图
        thresh： 阈值
        maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
        type：二值化操作的类型，包含以下5种类型： 
            cv2.THRESH_BINARY           超过阈值部分取maxval（最大值），否则取0
            cv2.THRESH_BINARY_INV       对THRESH_BINARY的反转
            cv2.THRESH_TRUNC            大于阈值部分设为阈值，否则不变
            cv2.THRESH_TOZERO           大于阈值部分不改变，否则设为0
            cv2.THRESH_TOZERO_INV       对THRESH_TOZERO的反转
        
'''
# img = cv2.imread('img1/cat.jpg', cv2.IMREAD_GRAYSCALE)
# ret1, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ret2, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# ret3, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# ret4, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# ret5, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
# res = np.hstack((thresh1, thresh2, thresh3, thresh4, thresh5))
# cv_show(res)

'''
    边缘检测-Canny:
        实现步骤：
           1、使用高斯滤波器，以平滑图像，滤除噪声。
           2、计算图像中每个像素点的梯度强度和方向。
           3、应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
           4、应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
           5、通过抑制孤立的弱边缘最终完成边缘检测。
        
        minVal：小于该值的像素点将会被丢弃，值越小提取越精细
        maxVal：达到最大值的点将会保留，处理为边界，值越大提取边界越模糊
        minVal < 像素点 < maxVal：连有边界则保留，否则则舍弃
'''
# img = cv2.imread('img1/car.png', cv2.IMREAD_GRAYSCALE)
# canny = cv2.Canny(img, 100, 250)
# cv_show(canny)

'''
    图像轮廓：cv2.findContours(img,mode,method)
        mode:轮廓检索模式
            RETR_EXTERNAL ：只检索最外面的轮廓；
            RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
            RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
            RETR_TREE(一般都用这个)：检索所有的轮廓，并重构嵌套轮廓的整个层次;
            
        method:轮廓逼近方法常用的两种
            CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
            CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
        
        注意：为了更高的准确率，使用二值图像。
        
        返回值说明:
            binary：传入的二值图像
            contours：得到的图像中所有轮廓的列表
            hierarchy：层级
            
    绘制轮廓：cv2.drawContours(image, contours, contourIdx, color, thickness)
        image：要绘制的图像
        contours：轮廓信息
        contourIdx：要绘制图像中第几个轮廓，-1表示绘制所有
        color：绘制轮廓使用的颜色
        thickness：绘制轮廓使用的线条粗细
'''
# img = cv2.imread('img1/contours.png')
# # 这里必须使用颜色转换一下
# gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # 将图像变为二值图像
# ret, thresh = cv2.threshold(gary, 127, 255, cv2.THRESH_BINARY)
# # 得到轮廓信息
# binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# # 这里copy出来一个新图像，不在原图上绘制轮廓信息
# draw_img = img.copy()
# res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
# cv_show(res)

'''
    图像轮廓-特征：
        面积：cv2.contourArea(cnt)
            cnt：要计算面积的特征
        周长：cv2.arcLength(cnt,True)
            cnt：要计算面积的特征
            True：表示闭合的，一般都为True
'''
# img = cv2.imread('img1/contours.png')
# gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gary, 127, 255, cv2.THRESH_BINARY)
# binary, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# area = cv2.contourArea(contours[0])
# print(area)
# arc = cv2.arcLength(contours[0], True)
# print(arc)

'''
    图像轮廓-轮廓相似： cv2.approxPolyDP(cnt,epsilon,True)
        当一个轮廓绘制出来有很多弯曲线的时候，可以使用轮廓相似将这些弯曲线尽可能变成直线
        首先在A,B两点的曲线上找出最高点C，然后计算这个最高点到A，B两点的直线距离，如果该距离小于
        我们传入的值，那么则使用直线代替曲线，反之将曲线进行拆分对A，C和B，C做同样A，B的操作，
        直到能够使用直线代替为止
        
        cnt：轮廓
        epsilon(一般使用周长的倍数)：指定一个比较值，当该值大于计算出的阈值，则使用直线代替
        原来的曲线，否则则使用二分法将曲线进行拆分，再次比较阈值
'''
# img = cv2.imread('img1/contours2.png')
# gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gary, 127, 255, cv2.THRESH_BINARY)
# binary, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# source_res = cv2.drawContours(img.copy(), contours, 0, (0, 255, 0), 2)
# cv_show(source_res)
# # 获取近似轮廓
# epsilon = 0.1 * cv2.arcLength(contours[0], True)
# poly = cv2.approxPolyDP(contours[0], epsilon, True)
# res = cv2.drawContours(img.copy(), [poly], -1, (0, 255, 0), 2)
# cv_show(res)

'''
    图像轮廓-边缘矩形：cv2.boundingRect(cnt)
        就是将提取出来的轮廓根据各个位置顶点外边套一层长方形
        boundingRect函数获取这个图形的外接矩形的四个坐标
        
        rectangle函数绘制外接矩形:
            img：绘制的图像
            pt1：矩形的顶点
            pt2：与pt1相对的矩形顶点
            color：绘制的颜色
            thickness：绘制线条的粗细
'''
# img = cv2.imread('img1/contours.png')
# gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gary, 127, 255, cv2.THRESH_BINARY)
# binary, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cnt = contours[0]
# x, y, w, h = cv2.boundingRect(cnt)
# res = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv_show(res)

'''
    图像轮廓-外接圆：cv2.boundingRect(cnt)
        就是将提取出来的轮廓根据各个位置顶点外边套一层圆形
        minEnclosingCircle函数获取这个图形的外接圆x和y轴的圆心、半径

        circle函数绘制外接圆:
            img：绘制的图像
            pt1：圆心
            pt2：半径
            color：绘制的颜色
            thickness：绘制线条的粗细
'''
# img = cv2.imread('img1/contours.png')
# gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gary, 127, 255, cv2.THRESH_BINARY)
# binary, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cnt = contours[0]
# (x, y), radius = cv2.minEnclosingCircle(cnt)
# center = (int(x), int(y))
# radius = int(radius)
# res = cv2.circle(img, center, radius, (0, 255, 0), 2)
# cv_show(res)

'''
    图像金字塔-高斯金字塔：将图片分层，每一层都是2的倍数组成金字塔
            向下采样即金字塔底部到顶部（缩小）：
                先做高斯内核卷积，然后将所有偶数列和行都去除
            向上采样即金字塔顶部到底部（放大）：
                将图像每个方向都扩大两倍，新增加的行和列以0填充，
                然后用先前大小的卷积核再做一次卷积操作
                
            pyrUp：上采样，扩大2倍
            pyrDown：下采样，缩小2倍
            
            注意：先上采样，在下采样，得到的结果图和原图是不相等的，因为会有像素丢失
'''
# img = cv2.imread('img1/AM.png')
# up = cv2.pyrUp(img)
# down = cv2.pyrDown(img)
# cv_show(up)
# cv_show(down)

'''
    图像金字塔-拉普拉斯金字塔：
        先得到一张原始图片，将这张图缩小再放大，得到结果，然后将原始图片减去先缩小再放大的图片
'''
img = cv2.imread('img1/AM.png')
down = cv2.pyrDown(img)
down_up = cv2.pyrUp(down)
l_1 = img - down_up
cv_show(l_1)
