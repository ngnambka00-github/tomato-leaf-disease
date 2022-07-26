import numpy as np
import cv2

from hydra import initialize, compose
from omegaconf import OmegaConf

with initialize(config_path="../config/"):
    data_cfg = compose(config_name="data_path")
data_cfg = OmegaConf.create(data_cfg)

# anh truyen vao la anh mau -> tach nen cach so 1
def remove_background(image):
    image = np.copy(image)

    # create hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # set lower and upper color limits
    low_val = (0,60,0)
    high_val = (179,255,255)
    # Threshold the HSV image 
    mask = cv2.inRange(hsv, low_val,high_val)
    # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((8,8),dtype=np.uint8))
    # apply mask to original image
    bg_remove_img = cv2.bitwise_and(image, image,mask=mask)

    return bg_remove_img

# segmentation anh dau vao -> tach nen cach so 2
def img_segmentation(rgb_img, hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)

    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)

    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)

    return final_result
    
# anh dau vao la anh xam
# bo loc sobel -> cat nguong de lay anh black-white
def sobel_edge_detection(image, blur_ksize=5, sobel_ksize=1, skipping_threshold=10):
    gray = np.copy(image)
    img_gaussian = cv2.GaussianBlur(gray,(blur_ksize,blur_ksize),0)

    sobelx64f = cv2.Sobel(img_gaussian,cv2.CV_64F,1,0,ksize=sobel_ksize)
    abs_sobel64f = np.absolute(sobelx64f)
    img_sobelx = np.uint8(abs_sobel64f)

    sobely64f = cv2.Sobel(img_gaussian,cv2.CV_64F,1,0,ksize=sobel_ksize)
    abs_sobel64f = np.absolute(sobely64f)
    img_sobely = np.uint8(abs_sobel64f)

    img_sobel = (img_sobelx + img_sobely)/2
    for i in range(img_sobel.shape[0]):
        for j in range(img_sobel.shape[1]):
            if img_sobel[i][j] < skipping_threshold:
                img_sobel[i][j] = 0
            else:
                img_sobel[i][j] = 255
    return img_sobel

# anh truyen vao la anh xam
# sobel co chuan hoa (tra ve gia tri 0 -> 255)
def sobel_edge_detection_2(image, blur_ksize=5, sobel_ksize=3):
    gray = np.copy(image)
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    # sobel algorthm use cv2.CV_64F
    sobelx64f = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    abs_sobel64f = np.absolute(sobelx64f)
    img_sobelx = np.uint8(abs_sobel64f)

    sobely64f = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    abs_sobel64f = np.absolute(sobely64f)
    img_sobely = np.uint8(abs_sobel64f)
    
    img_sobel = (img_sobelx + img_sobely)
    
    return img_sobel

# anh truyen vao la anh xam
# bo loc prewitt -> cat nguong de lay anh black-white
def prewitt_edge_detection(image, blur_ksize = 5, skipping_threshold=30):
    gray = np.copy(image)
    img_gaussian = cv2.GaussianBlur(gray,(blur_ksize,blur_ksize),0)

    #prewitt
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    img_prewitt1 = (img_prewittx + img_prewitty)/2
    
    kernelx2 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    kernely2 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    img_prewittx2 = cv2.filter2D(img_gaussian, -1, kernelx2)
    img_prewitty2 = cv2.filter2D(img_gaussian, -1, kernely2)
    img_prewitt2 = (img_prewittx2 + img_prewitty2)/2
    
    img_prewitt = (img_prewitt1 + img_prewitt2)/2
    for i in range(img_prewitt.shape[0]):
        for j in range(img_prewitt.shape[1]):
            if img_prewitt[i][j] < skipping_threshold:
                img_prewitt[i][j] = 0
            else:
                img_prewitt[i][j] = 255
    return img_prewitt

# anh truyen vao la anh xam
# tinh egde theo phuong phap canny
def canny_edge_detection(img, blur_ksize=5, threshold1=100, threshold2=200, skipping_threshold=30):
    gray = np.copy(img)
    img_gaussian = cv2.GaussianBlur(gray,(blur_ksize,blur_ksize),0)
    img_canny = cv2.Canny(img_gaussian,threshold1,threshold2)
#     for i in range(img_canny.shape[0]):
#         for j in range(img_canny.shape[1]):
#             if img_canny[i][j] < skipping_threshold:
#                 img_canny[i][j] = 0
    return img_canny

# anh truyen vao la anh xam
# bo loc lapacian -> co duoc chuan hoa
def lapacian_detection(image):
    img = np.copy(image)
    lap_img = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    abs_img = np.absolute(lap_img)
    result = np.uint8(abs_img)
    return result

# remove noise (gausian) -> lapacian filter -> normailize in range(0, 255)
def lapacian_detection_2(image):
    img = np.copy(image)
    g_blur = cv2.GaussianBlur(img, (5, 5), 0)
    lap_img = cv2.Laplacian(g_blur, cv2.CV_64F, ksize=3)
    abs_img = np.absolute(lap_img)
    result = np.uint8(abs_img)
    return result

# segment image -> tach nen cho anh
def img_segmentation(rgb_img, hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    

    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)

    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)

    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    
    return final_result

# step xu ly du lieu
# don gian chi co remove bg -> convert to gray -> remove noise by gau
# Nen remove noise -> roi moi sharpen 
def step_preprocessing(img):
    # remove background
    rm_bg_img = remove_background(img)

    # convert to gray image
    gray_img = cv2.cvtColor(rm_bg_img, cv2.COLOR_BGR2GRAY)

    # sharpen image
    sharpen_img = lapacian_detection(gray_img)

    # remove noise
    remove_noise_img = cv2.GaussianBlur(sharpen_img, (5, 5), 0)
    
    # fuzzy mean image
    # fcm_img = fcm_image(remove_noise_img, n_cluster=FCM_CLUSTER)

    return remove_noise_img
