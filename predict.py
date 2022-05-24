import cv2
import matplotlib.pyplot
import numpy as np
from numpy.linalg import norm
import sys
import os
import json
import matplotlib as plt

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H

        .
    """

    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):#巴特沃斯滤波器
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):#高斯滤波
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0]) ** 2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):#傅里叶别换
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b * H) * I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):#同态滤波
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) != 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'external':
            print('external')
            if len(H.shape) != 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt)) - 1
        return np.uint8(I)


# End of class HomomorphicFilter
# 读取图片文件
def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


# 根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


# 来自opencv的sample，用于svm训练
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


# 不能保证包括所有省份
provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "靑",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        # 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
        f = open('config.js')
        j = json.load(f)
        for c in j["config"]:
            if c["open"]:
                self.cfg = c.copy()
                break
        else:
            raise RuntimeError('没有设置有效配置参数')
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)

    def __del__(self):
        self.__save_traindata()

    def __save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.modelchinese.save("svmchinese.dat")

    def __accurate_place(self, card_img_hsv, limit1, limit2, color):
        row_num, col_num = card_img_hsv.shape[:2]
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        col_num_limit = self.cfg["col_num_limit"]
        row_num_limit = self.cfg["row_num_limit"]
        col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
        for i in range(row_num):
            count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > col_num_limit:
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
        for j in range(col_num):
            count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > row_num - row_num_limit:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
        return xl, xr, yh, yl

    def train_svm(self):
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("train/chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(root_int)

            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            chars_label = np.array(chars_label)
            self.model.train(chars_train, chars_label)
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
            for root, dirs, files in os.walk("train/charsChinese"):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(index)
            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            chars_label = np.array(chars_label)
            self.modelchinese.train(chars_train, chars_label)

    def __imgshow_and_key_detect(self, window_name, img):#按a进行下一步
        cv2.imshow(window_name, img)
        while True:
            key = cv2.waitKey()
            if key == ord('a'):
                break

    def __gamma_transform(self, img, gamma):#gamma变换
        Dm = 255
        img = np.power(img / Dm, gamma)
        img = np.array(Dm * img, dtype=np.uint8)
        return img

    def __s_transform(self, alpha, img):#S变换（未使用）
        img = 255 / 2 * (1 + 1 / np.tan(alpha * 3.1415926) * np.tan(
            alpha * 3.1415926 * (np.array(img, dtype=np.float32) / 255 - 1 / 2)))
        img = np.array(img, dtype=np.uint8)
        return img

    def __show_plt(self, img):#显示灰度直方图
        plt.pyplot.hist(img.ravel(), 256, [0, 256])
        plt.pyplot.show()

    def __resize(self, car_pic, resize_rate=1):#分辨率调整
        if type(car_pic) == type(""):
            img = imreadex(car_pic)
        else:
            img = car_pic
        pic_hight, pic_width = img.shape[:2]
        if pic_width > MAX_WIDTH:
            pic_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * pic_rate)), interpolation=cv2.INTER_LANCZOS4)
            pic_hight, pic_width = img.shape[:2]

        if resize_rate != 1:
            img = cv2.resize(img, (int(pic_width * resize_rate), int(pic_hight * resize_rate)),
                             interpolation=cv2.INTER_LANCZOS4)
            pic_hight, pic_width = img.shape[:2]
            print("h,w:", pic_hight, pic_width)
        return img, pic_hight, pic_width

    def __homofilter(self,img,a=0.75, b=1.25,para1=30,para2=2):
        homo_filter = HomomorphicFilter(a=a, b=b)  # a = 0.75, b = 1.25
        img = homo_filter.filter(I=img, filter_params=[para1, para2])
        return img

    def __dark_or_not(self,img,dark_threshold,prop_threshold,mode):#1为判断暗，2为判断亮
        # 把图片转换为灰度图
        # 获取灰度图矩阵的行数和列数
        r, c = img.shape[:2]
        dark_sum = 0 # 偏暗的像素 初始化为0个
        dark_prop = 0  # 偏暗像素所占比例初始化为0
        pixels_sum = r * c  # 整个灰度图的像素个数为r*c
        # 遍历灰度图的所有像素
        if mode==1:
            for row in img:
                for colum in row:
                    if int(colum) < dark_threshold:  # 人为设置的超参数,表示0~39的灰度值为暗
                        dark_sum += 1
            dark_prop = dark_sum / pixels_sum
            print("这么多黑的像素点:" + str(dark_sum))
            print("一共这么多像素点:" + str(pixels_sum))
            print("暗的像素点所占比例:" + str(dark_prop))
        elif mode==2:
            for row in img:
                for colum in row:
                    if int(colum) > dark_threshold:  # 人为设置的超参数,表示0~39的灰度值为暗
                        dark_sum += 1
            dark_prop = dark_sum / pixels_sum
            print("这么多亮的像素点:" + str(dark_sum))
            print("一共这么多像素点:" + str(pixels_sum))
            print("亮的像素点所占比例:" + str(dark_prop))
        if dark_prop >= prop_threshold:  # 人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
            print("满足",prop_threshold*100,"%","的灰度在",dark_threshold,"下的要求")
            return True
        else:
            print("不满足",prop_threshold*100,"%","的灰度在",dark_threshold,"下的要求")
            return False

    def predict(self, car_pic, resize_rate=1):
        img, pic_hight, pic_width = self.__resize(car_pic)  # 重定尺寸

        blur = self.cfg["blur"]
        if blur > 0:
            img = cv2.GaussianBlur(img, (blur, blur), 0)  # 高斯去噪声
        oldimg = img

        self.__imgshow_and_key_detect("RGB_after_gauss", img) #显示

        if self.__dark_or_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),50,0.75,1):#很暗
            self.__show_plt(img)  # 显示灰度直方图
            img = self.__gamma_transform(img, 0.25)
            img = cv2.GaussianBlur(img, (blur, blur), 0)
            self.__imgshow_and_key_detect("gamma", img)  # 显示
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.__dark_or_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 75, 0.65,1):#较暗
            self.__show_plt(img)  # 显示灰度直方图
            img = self.__gamma_transform(img, 0.5)
            img = cv2.GaussianBlur(img, (blur, blur), 0)
            self.__imgshow_and_key_detect("gamma", img)  # 显示
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.__dark_or_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200, 0.75, 2):  # 很亮
            self.__show_plt(img)  # 显示灰度直方图
            img = self.__gamma_transform(img, 1.5)
            img = cv2.GaussianBlur(img, (blur, blur), 0)
            self.__imgshow_and_key_detect("gamma", img)  # 显示
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.__dark_or_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 175, 0.65,2):#较亮
            self.__show_plt(img)  # 显示灰度直方图
            img = self.__gamma_transform(img, 1.25)
            img = cv2.GaussianBlur(img, (blur, blur), 0)
            self.__imgshow_and_key_detect("gamma", img)  # 显示
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            self.__show_plt(img)  # 显示灰度直方图
            img = self.__gamma_transform(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)  # 直方图均衡
            img = cv2.GaussianBlur(img, (blur, blur), 0)
            self.__imgshow_and_key_detect("equality", img)  # 显示
        self.__homofilter(img)
        self.__imgshow_and_key_detect("grey_after_homo", img)  # 显示
        self.__show_plt(img)#显示灰度直方图

        # equ = cv2.equalizeHist(img)#直方图均衡（未使用）
        # img = np.hstack((img, equ))#原图与均衡后的图叠加（未使用）

        # 去掉图像中不会是车牌的区域
        kernel = np.ones((20, 20), np.uint8)
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)

        # 找到图像边缘
        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_edge = cv2.Canny(img_thresh, 100, 200)
        self.__imgshow_and_key_detect("canny", img_thresh)
        # 使用开运算和闭运算让图像边缘成为一个整体
        kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
        self.__imgshow_and_key_detect("open_close", img_edge2)

        # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
        try:
            contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
        # 一一排除不是车牌的矩形区域
        car_contours = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            area_width, area_height = rect[1]
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if wh_ratio > 2 and wh_ratio < 5:
                car_contours.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
            try:
                oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
            except:
                print("框不出来")
            self.__imgshow_and_key_detect("oldimg",oldimg)

        print("这么多轮廓：", len(car_contours))

        card_imgs = []
        # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
        for rect in car_contours:
            if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
                angle = 1
            else:
                angle = rect[2]
            rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除

            box = cv2.boxPoints(rect)
            heigth_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_hight]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if heigth_point[1] < point[1]:
                    heigth_point = point
                if right_point[0] < point[0]:
                    right_point = point

            if left_point[1] <= right_point[1]:  # 正角度
                new_right_point = [right_point[0], heigth_point[1]]
                pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, heigth_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                point_limit(new_right_point)
                point_limit(heigth_point)
                point_limit(left_point)
                card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
                card_imgs.append(card_img)
                new_card_img = card_img
                self.__imgshow_and_key_detect("card",card_img)
            elif left_point[1] > right_point[1]:  # 负角度
                new_left_point = [left_point[0], heigth_point[1]]
                pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, heigth_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                point_limit(right_point)
                point_limit(heigth_point)
                point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
                card_imgs.append(card_img)
                new_card_img = card_img
                self.__imgshow_and_key_detect("card",card_img)
        # 开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
        colors = []
        for card_index, card_img in enumerate(card_imgs):
            green = yello = blue = black = white = 0
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            self.__imgshow_and_key_detect("hsv",card_img_hsv)
            # 有转换失败的可能，原因来自于上面矫正矩形出错
            if card_img_hsv is None:
                continue
            row_num, col_num = card_img_hsv.shape[:2]
            card_img_count = row_num * col_num

            for i in range(row_num):
                for j in range(col_num):
                    H = card_img_hsv.item(i, j, 0)
                    S = card_img_hsv.item(i, j, 1)
                    V = card_img_hsv.item(i, j, 2)
                    if 11 < H <= 34 and S > 34:  # 图片分辨率调整
                        yello += 1
                    elif 35 < H <= 99 and S > 34:  # 图片分辨率调整
                        green += 1
                    elif 99 < H <= 124 and S > 34:  # 图片分辨率调整
                        blue += 1

                    if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                        black += 1
                    elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                        white += 1
            color = "no"

            limit1 = limit2 = 0
            if yello * 2 >= card_img_count:
                color = "yello"
                limit1 = 11
                limit2 = 34  # 有的图片有色偏偏绿
            elif green * 2 >= card_img_count:
                color = "green"
                limit1 = 35
                limit2 = 99
            elif blue * 2 >= card_img_count:
                color = "blue"
                limit1 = 100
                limit2 = 124  # 有的图片有色偏偏紫
            elif black + white >= card_img_count * 0.7:  # TODO
                color = "bw"
            print(color)
            colors.append(color)
            print(blue, green, yello, black, white, card_img_count)
            if limit1 == 0:
                continue
            # 以上为确定车牌颜色
            # 以下为根据车牌颜色再定位，缩小边缘非车牌边界
            xl, xr, yh, yl = self.__accurate_place(card_img_hsv, limit1, limit2, color)
            if yl == yh and xl == xr:
                continue
            need_accurate = False
            if yl >= yh:
                yl = 0
                yh = row_num
                need_accurate = True
            if xl >= xr:
                xl = 0
                xr = col_num
                need_accurate = True
            card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
                                                                                                           yl - (
                                                                                                                   yh - yl) // 4:yh,
                                                                                                           xl:xr]
            if need_accurate:  # 可能x或y方向未缩小，需要再试一次
                card_img = card_imgs[card_index]
                card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
                xl, xr, yh, yl = self.__accurate_place(card_img_hsv, limit1, limit2, color)
                if yl == yh and xl == xr:
                    continue
                if yl >= yh:
                    yl = 0
                    yh = row_num
                if xl >= xr:
                    xl = 0
                    xr = col_num
            card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
                                                                                                           yl - (
                                                                                                                   yh - yl) // 4:yh,
                                                                                                           xl:xr]
        # 以上为车牌定位
        # 以下为识别车牌中的字符
        predict_result = []
        roi = None
        card_color = None
        for i, color in enumerate(colors):
            if color in ("blue", "yello", "green"):
                # card_img = new_card_img
                card_img = card_imgs[i]
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # 查找水平直方图波峰
                x_histogram = np.sum(gray_img, axis=1)
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = find_waves(x_threshold, x_histogram)
                if len(wave_peaks) == 0:
                    print("根本没波峰")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

                wave_peaks = find_waves(y_threshold, y_histogram)

                # for wave in wave_peaks:
                #	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
                # 车牌字符数应大于6
                if len(wave_peaks) <= 6:
                    print("peak less 1:", len(wave_peaks))
                    continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                # 去除车牌上的分隔点
                point = wave_peaks[2]
                if point[1] - point[0] < max_wave_dis / 3:
                    point_img = gray_img[:, point[0]:point[1]]
                    if np.mean(point_img) < 255 / 5:
                        wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    print("peak less 2:", len(wave_peaks))
                    continue
                part_cards = seperate_card(gray_img, wave_peaks)
                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉
                    if np.mean(part_card) < 255 / 5:
                        print("a point")
                        continue
                    part_card_old = part_card
                    # w = abs(part_card.shape[1] - SZ)//2
                    w = part_card.shape[1] // 3
                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    # cv2.imwrite("u.jpg", part_card)
                    # part_card = deskew(part_card)
                    part_card = preprocess_hog([part_card])
                    if i == 0:
                        resp = self.modelchinese.predict(part_card)
                        charactor = provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        charactor = chr(int(resp[0]))
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if charactor == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 8:  # 1太细，认为是边缘
                            print(part_card_old.shape)
                            continue
                    predict_result.append(charactor)
                roi = card_img
                card_color = color
                break

        return predict_result, roi, card_color  # 返回识别到的字符、定位的车牌图像、车牌颜色


if __name__ == '__main__':
    c = CardPredictor()
    c.train_svm()
    file = input("请输入文件路径：")
    predict_result, roi, card_color = c.predict(file)
    print(predict_result, card_color)
    try:
        cv2.imshow("final",roi)
        cv2.waitKey()
    except:
        print("识别不出来！")