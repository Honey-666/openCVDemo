# @FileName：Demo2.py
# @Description：
# @Author：dyh
# @Time：2023/4/3 21:08
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np
from matplotlib import pyplot as plt


def cv_show(im):
    cv2.imshow('demo', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
    模板匹配：cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
        给定两张图片，其中一张为模板图，将另一张切分成相同大小的图片和模板图片比较找出最相似的那一张
        
        image：图片
        templ：模板图片
        method：匹配方法，建议都是用归一化的方法，这样匹配的更精准
            TM_SQDIFF：计算平方不同，计算出来的值越小，越相关
            TM_CCORR：计算相关性，计算出来的值越大，越相关
            TM_CCOEFF：计算相关系数，计算出来的值越大，越相关
            TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
            TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
            TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关
            
    cv2.minMaxLoc(res):
        可以将切分后每张图片的相似度进行比较，返回最大和最小的和最大最小位置的坐标，
        注意此坐标都是表示的是左上角的坐标
'''

# template = cv2.imread('img2/face.jpg', 0)
# h, w = template.shape[:2]
# img = cv2.imread('img2/lena.jpg', 0)
# res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
# # 如果是平方差匹配TM_CCORR、TM_CCOEFF或归一化平方差匹配TM_CCORR_NORMED、TM_CCOEFF_NORMED，取最大值
# # 这里高使用加法是因为高度从最上面是0开始的，所以使用加法
# bottom_right = (min_loc[0] + w, min_loc[1] + h)
# # 将得到的图片画出来
# res = cv2.rectangle(img.copy(), min_loc, bottom_right, 255, 2)
# cv_show(res)
# --------------------------取最大值情况----------------------
# template = cv2.imread('img2/face.jpg', 0)
# h, w = template.shape[:2]
# img = cv2.imread('img2/lena.jpg', 0)
# res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
# # 如果是平方差匹配TM_CCORR、TM_CCOEFF或归一化平方差匹配TM_CCORR_NORMED、TM_CCOEFF_NORMED，取最大值
# # 这里高使用加法是因为高度从最上面是0开始的，所以使用加法
# bottom_right = (max_loc[0] + w, max_loc[1] + h)
# # 将得到的图片画出来
# res = cv2.rectangle(img.copy(), max_loc, bottom_right, 255, 2)
# cv_show(res)
# --------------------------一张图片存在多个和模板相匹配的情况----------------------
# template = cv2.imread('img2/mario_coin.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('img2/mario.jpg')
# img_gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# h, w = template.shape
# res = cv2.matchTemplate(img_gary, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# # 将多个结果中大于%80的坐标全部匹配出来
# loc = np.where(res >= threshold)
# '''
#     loc 是 ([1,2,3],[4,5,6]) 这样一种结构，
#     [::-1]则是对数组的一个反转
#     当使用了zip(loc)后就会变成[(1,4),(2,5),(3,6)]
#     当在为zip()里面加上*后zip(*loc)就会变成(1,4),(2,5),(3,6)
# '''
# for pt in zip(*loc[::-1]):  #经过80%的过滤得到的都是最大的所以用TM_CCOEFF_NORMED
#     bottom_right = (pt[0] + w, pt[1] + h)
#     cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)
# cv_show(img)

'''
    傅里叶变换：
        高频：变化剧烈的灰度分量，例如边界区域
        低频：变化缓慢的灰度分量，例如非边界区域，距离中间越近，频率越低
        
        
        滤波：
            低通滤波器：只保留低频，会使得图像模糊
            高通滤波器：只保留高频，会使得图像细节增强
            
        opencv中主要就是cv2.dft() 变换和cv2.idft() 逆变换，通常展示实用，输入图像需要先转换成np.float32 格式。
        得到的结果中频率为0的部分会在左上角，通常要转换到中心位置，可以通过shift变换来实现。
        cv2.dft()返回的结果是双通道的（实部，虚部），通常还需要转换成图像格式才能展示（0,255）。
'''
# img = cv2.imread('img2/lena.jpg', 0)
# img_float = np.float32(img) #转为32
# dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT) # 得到变换结果
# dft_shift = np.fft.fftshift(dft) # shift变换
# # 得到灰度图能表示的形式
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
# # 拼接原图和结果图展示
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# ------------------低通滤波器------------
# img = cv2.imread('img2/lena.jpg', 0)
# img_float = np.float32(img)  # 转为32
# dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)  # 得到变换结果
# dft_shift = np.fft.fftshift(dft)  # shift变换
# h, w = img.shape
# crow, ccol = int(h / 2), int(w / 2)  # 中心位置
# # 低通滤波
# mask = np.zeros((h, w, 2), np.uint8)  # 初始化都为0
# mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1  # 将中心区域变为1
#
# # IDFT 将转换的傅里叶，再转为原来的图片，逆操作
# fshift = dft_shift * mask #将掩码和结果结合在一起，为1则保留，不为1则丢弃
# f_ishift = np.fft.ifftshift(fshift) #将中心点在转换为左上角
# img_back = cv2.idft(f_ishift) #逆操作
# img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]) #双通道变换
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
# plt.title('Result'), plt.xticks([]), plt.yticks([])
# plt.show()
# ------------------高通滤波器------------
# img = cv2.imread('img2/lena.jpg', 0)
# img_float32 = np.float32(img)
# dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
#
# h, w = img.shape
# crow, ccol = int(h / 2), int(w / 2)  # 中心位置
#
# # 高通滤波
# mask = np.ones((h, w, 2), np.uint8)  # 这里都初始化为1
# mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0  # 中间区域变为0
#
# # IDFT
# fshift = dft_shift * mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv2.idft(f_ishift)
# img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
#
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img_back, cmap='gray')
# plt.title('Result'), plt.xticks([]), plt.yticks([])
#
# plt.show()

'''
    直方图：cv2.calcHist(images,channels,mask,histSize,ranges)
        就是将图片的每个像素都拿出来，然后做一个汇总，以x，y轴的线型方式展现出来
        
        images: 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应 用中括号 [] 括来例如[img]
        channels: 同样用中括号括来它会告函数我们统幅图 像的直方图。如果入图像是灰度图它的值就是 [0]如果是彩色图像 的传入的参数可以是 [0][1][2] 它们分别对应着 BGR。
        mask: 掩模图像。统整幅图像的直方图就把它为 None。但是如 果你想统图像某一分的直方图的你就制作一个掩模图像并 使用它。
        histSize:BIN 的数目。也应用中括号括来
        ranges: 像素值范围常为 [0256]
'''
# ----------------无mask
# img = cv2.imread('img1/cat.jpg', 0)
# cv2.calcHist([img], [0], None, [256], [0, 256])  # 无mask
# plt.hist(img.ravel(), 256)
# plt.show()
# ---------------有mask
# 制造mask
# mask = np.zeros(img.shape[:2], np.uint8)  # 对这个图片的宽高所有像素都初始化为0
# mask[100:300, 100:400] = 255  # 将部分区域像素变为255
# img = cv2.imread('img1/cat.jpg', 0)
# masked_img = cv2.bitwise_and(img, img, mask=mask)  # 进行与操作，255的部分会保留原图
# cv_show(masked_img)
# hist_full = cv2.calcHist([img], [0], None, [256], [0, 256]) #无mask
# hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256]) # 有mask
# #图片显示
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(mask, 'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0, 256])
# plt.show()

'''
    直方图均衡化:就是将像素更平均，通俗的说就是将细高变成矮胖
        均衡化会丢失一些细节，并不是所有的图片均衡化后效果会亮
'''
# img = cv2.imread('img2/clahe.jpg', 0)  # 0表示灰度图 #clahe
# plt.hist(img.ravel(), 256)
# plt.show()
# # 均衡化函数
# equ = cv2.equalizeHist(img)
# plt.hist(equ.ravel(), 256)
# plt.show()
# res = np.hstack((img, equ))
# cv_show(res)
'''
    自适应直方图均衡化:
        将图片分割成不同的块，对每个块进行均衡化，然后在拼接这些分割的图片
'''
#定义每个块大小，和均衡化处理
img = cv2.imread('img2/clahe.jpg', 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#自适应直方图均衡化
res_clahe = clahe.apply(img)
# 均衡化函数
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ, res_clahe))
cv_show(res)
