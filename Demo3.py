# @FileName：Demo3.py
# @Description：
# @Author：dyh
# @Time：2023/4/13 21:20
# @Website：www.xxx.com
# @Version：V1.0
import cv2


def cv_show(im):
    cv2.imshow("demo", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
    角点检测：cv2.cornerHarris()
        参数：
            img： 数据类型为 ﬂoat32 的入图像
            blockSize： 角点检测中指定区域的大小
            ksize： Sobel求导中使用的窗口大小
            k： 取值参数为 [0,04,0.06]
'''
# img = cv2.imread('img4/test_1.jpg')
# gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# harr = cv2.cornerHarris(gary, 2, 3, 0.04)
# img[harr > 0.09 * harr.max()] = [0, 0, 255]
# cv_show(img)

'''
    图像特征-SIFT算法：sift = cv2.xfeatures2d.SIFT_create()
        
'''
img = cv2.imread('img4/test_1.jpg')
gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
keyPoint = sift.detect(gary, None) #获取关键点
img = cv2.drawKeypoints(gary, keyPoint, img) #画出关键点
cv_show(img)
# 计算特征
kp, des = sift.compute(gary, keyPoint) #计算特征向量
print(des.shape)
