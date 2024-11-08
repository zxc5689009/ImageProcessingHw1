from PyQt5.QtCore import QByteArray, QBuffer, QIODevice
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from ipui import Ui_MainWindow
import sys
import cv2
import numpy as np
sigma=1.5
sigma1=40
def Load_Image():
    global Img_name, Ori_img
    Img_name, _ = QFileDialog.getOpenFileName(None, 'Select a file', '', 'All Files (*);;Text Files (*.txt)')
    print(Img_name)
    tmp = f'<img src = "{Img_name}" width = "200" height = "160"/>'
    UI.textEdit.setHtml(tmp)

def img_to_base64(pixmap):
    byte_array = QByteArray()
    buffer = QBuffer(byte_array)
    buffer.open(QIODevice.WriteOnly)
    pixmap.save(buffer, 'PNG')
    base64_data = byte_array.toBase64().data().decode()
    return base64_data

def Average_Filter(img_tmp, kernel):
    img=img_tmp.copy()
    blurred_image = cv2.blur(img, (kernel, kernel))
    height, width, channel = blurred_image.shape
    bytes_per_line = 3 * width
    qt_image = QImage(blurred_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_image)
    tmp2 = f'<img src="data:image/png;base64,{img_to_base64(pixmap)}" width = "200" height = "160" />'
    UI.textEdit2.setHtml(tmp2)
    UI.Label2.setText("1(a)Average filter")

def Median_Filter(img_tmp, kernel):
    img=img_tmp.copy()
    median_filter_image = cv2.medianBlur(img, kernel)
    height, width, channel = median_filter_image.shape
    bytes_per_line = 3 * width
    qt_image = QImage(median_filter_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_image)
    tmp3 = f'<img src="data:image/png;base64,{img_to_base64(pixmap)}" width = "200" height = "160" />'
    UI.textEdit3.setHtml(tmp3)
    UI.Label3.setText("1(a)Median filter")

def Fourier_transform_Filter(img_tmp):
    img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-40:crow+40, ccol-40:ccol+40] = 1
    fshift_masked = fshift * mask

    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back_normalized = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
    height, width = img_back_normalized.shape
    bytesPerLine = width
    qt_image = QImage(img_back_normalized.data, width, height, bytesPerLine, QImage.Format_Grayscale8)  
    pixmap = QPixmap.fromImage(qt_image)

    tmp4 = f'<img src="data:image/png;base64,{img_to_base64(pixmap)}"width = "200" height = "160" />'
    UI.textEdit4.setHtml(tmp4)
    UI.Label4.setText("1(b) Fourier transform")

def Smooth_Filter():
    Ori_img = cv2.imread(Img_name)
    global kernel_median
    global kernel_average
    #print(Ori_img.shape)
    #kernel_average+=2
    #kernel_median+=2
    Average_Filter(Ori_img, kernel_average)
    Median_Filter(Ori_img,kernel_median)
    Fourier_transform_Filter(Ori_img)

def Sobel_mask(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)
    sharpened = cv2.addWeighted(img,1, magnitude,1, 0)
    height, width, channel = sharpened.shape
    bytes_per_line = 3 * width
    qt_image = QImage(sharpened.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_image)
    tmp3 = f'<img src="data:image/png;base64,{img_to_base64(pixmap)}" width = "200" height = "160" />'
    UI.textEdit3.setHtml(tmp3)
    UI.Label3.setText("2(a)Sobel mask")

def Fourier_transform_sharp(img_tmp):
    img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY) 
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back_normalized = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
    sharpened = cv2.addWeighted(img,1,img_back_normalized,1, 0)
    height, width = img_back_normalized.shape
    bytesPerLine = width
    qt_image = QImage(sharpened.data, width, height, bytesPerLine, QImage.Format_Grayscale8)  
    pixmap = QPixmap.fromImage(qt_image)

    tmp4 = f'<img src="data:image/png;base64,{img_to_base64(pixmap)}"width = "200" height = "160" />'
    UI.textEdit4.setHtml(tmp4)
    UI.Label4.setText("2(b) Fourier transform")
    
def sharp():
    Ori_img = cv2.imread(Img_name)
    Sobel_mask(Ori_img)
    Fourier_transform_sharp(Ori_img)
    UI.textEdit2.setHtml("")
    UI.Label2.setText("No use")

def create_gaussian_mask(size, sigma):
    m, n = [(ss-1.)/2. for ss in size]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def apply_gaussian_filter(image, mask):
    return cv2.filter2D(image, -1, mask)
def Gaussian():
    global sigma
    #sigma+=0.5#1.5 OK
    print(f'sigma={sigma}')
    Ori_img_tmp = cv2.imread(Img_name)
    Ori_img = cv2.cvtColor(Ori_img_tmp, cv2.COLOR_BGR2GRAY)
    gaussian_mask = create_gaussian_mask((5, 5), sigma)
    result = apply_gaussian_filter(Ori_img,gaussian_mask)
    img_back_normalized = np.uint8(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX))


    height, width = img_back_normalized.shape
    bytesPerLine = width
    qt_image = QImage(img_back_normalized.data, width, height, bytesPerLine, QImage.Format_Grayscale8)  
    pixmap = QPixmap.fromImage(qt_image)

    tmp2 = f'<img src="data:image/png;base64,{img_to_base64(pixmap)}"width = "200" height = "160" />'
    UI.textEdit2.setHtml(tmp2)
    UI.Label2.setText("Result")  
    UI.textEdit3.setHtml("")
    UI.Label3.setText("No use")  
    UI.textEdit4.setHtml("")
    UI.Label4.setText("No use")    

def create_gaussian_lowpass_filter(shape, sigma):
    m, n = shape
    y, x = np.ogrid[-m//2:m//2, -n//2:n//2]
    filter_mask = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return filter_mask / filter_mask.sum()

def apply_fourier_lowpass_filter(image, sigma):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    rows, cols = gray_image.shape
    lowpass = create_gaussian_lowpass_filter((rows, cols), sigma)
    f_shift_filtered = f_shift * lowpass
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

def Lowerpass():
    global sigma1#27 OK
    #sigma1+=10
    print(f'sigma1={sigma1}')
    Ori_img_tmp = cv2.imread(Img_name)    
    result = apply_fourier_lowpass_filter(Ori_img_tmp,sigma1)
    img_back_normalized = np.uint8(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX))
    height, width = img_back_normalized.shape
    bytesPerLine = width
    qt_image = QImage(img_back_normalized.data, width, height, bytesPerLine, QImage.Format_Grayscale8)  
    pixmap = QPixmap.fromImage(qt_image)

    tmp2= f'<img src="data:image/png;base64,{img_to_base64(pixmap)}"width = "200" height = "160" />'
    UI.textEdit2.setHtml(tmp2)
    UI.Label2.setText("Result")
    UI.textEdit3.setHtml("")
    UI.Label3.setText("No use")    
    UI.textEdit4.setHtml("")
    UI.Label4.setText("No use")  
def main():
    global UI
    global kernel_average
    global kernel_median
    kernel_average = 5
    kernel_median  = 5
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    UI = Ui_MainWindow()
    UI.setupUi(MainWindow)
    UI.Load_Image.clicked.connect(Load_Image)
    UI.Smooth_Filter.clicked.connect(Smooth_Filter)
    UI.Sharp.clicked.connect(sharp)
    UI.Gaussian.clicked.connect(Gaussian)
    UI.Lowerpass.clicked.connect(Lowerpass)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
