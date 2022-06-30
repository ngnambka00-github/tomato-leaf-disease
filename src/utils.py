from cProfile import label
from operator import mod
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from fcmeans import FCM
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from hydra import initialize, compose
from omegaconf import OmegaConf

import cv2
import os
import os
import glob
import gc

from keras.preprocessing.image import ImageDataGenerator

with initialize(config_path="../config/"):
    data_cfg = compose(config_name="data_path")
data_cfg = OmegaConf.create(data_cfg)

datagen = ImageDataGenerator(
    fill_mode='constant',    # Tự động thêm các giá trị 0
    rotation_range=90, 
    zoom_range=[0.7, 1.0], 
    horizontal_flip=True, 
    vertical_flip=True, 
    brightness_range=[0.7,1.3], 
    width_shift_range=0.2, height_shift_range=0.2)

def generator_image(image, method_transform):
    data = np.copy(image)

    # expand dimension to one sample
    samples = np.expand_dims(data, 0)

    # create image data augmentation generator
    datagen = method_transform

    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    new_image = it.next()[0].astype('uint8')

    # save new generator image
    return new_image

# anh truyen vao la anh mau
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
    

# anh dau vao la anh xam
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
def lapacian_detection(image):
    img = np.copy(image)
    lap_img = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    abs_img = np.absolute(lap_img)
    result = np.uint8(abs_img)
    return result

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

# export image to features
def export_feature_from_folder(data_path, columns_name, feature_extraction_methods, export_data_path=None):

    # encoding label to int value 
    subdirs = os.listdir(data_path)
    label_encode = {key: index for index, key in enumerate(subdirs)}

    df = pd.DataFrame(columns=columns_name)

    for subdir in os.listdir(data_path):
        
        file_names = glob.glob(f"{os.path.join(data_path, subdir)}/**")
        len_file = len(file_names)
        
        for i in tqdm(range(len_file), desc=f"{subdir}"):
            path_image = file_names[i]

            # đọc ảnh
            image = cv2.imread(path_image)

            # step image
            step_image = step_preprocessing(image)

            # cac buoc trich dac trung
            total_features = np.array([])
            for method in feature_extraction_methods:
                total_features = np.concatenate((total_features, method(step_image)))
            total_features = np.concatenate((total_features, [label_encode[subdir]]))

            # Có thể xử lý dữ liệu trích xuất đặc trưng đưa vào file data
            df = df.append(
                { key: value for key, value in zip(columns_name, total_features) }, 
                ignore_index=True
            )

            if i % 100 == 0: 
                gc.collect()

    if export_data_path: 
        df.to_csv(export_data_path, index=False)
        
    return df

def split_data(data_path): 
    data = pd.read_csv(data_path)
    
    # chon cot features va cot labels
    features = np.array(data.iloc[:, :-1].values)
    labels = np.array(data.iloc[:, -1].values)

    # split data train va test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # nomalize du lieu
    normalize = StandardScaler()
    # normalize = StandardScaler()
    normalize.fit(X_train)
    X_train = normalize.transform(X_train)
    X_test = normalize.transform(X_test)

    # Xu ly imbalance data
    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return (X_train, y_train), (X_test, y_test)

# Train single model
def train_single_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predicts = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicts)
    precision = precision_score(y_test, predicts, average='macro')
    recall = recall_score(y_test, predicts, average='macro')
    f1 = f1_score(y_test, predicts, average='macro')
    return predicts, accuracy,precision, recall, f1

# Train nhieu model va chon ra model co do chinh xac cao nhat

def train_test_model_classification(models, X_train, y_train, X_test, y_test):
    print("TRAINING PROCESSING")

    log_cols=["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)

    best_accuracy = 0
    best_model = None
    best_predict = None
    for model in models: 
        preds, acc, precision, recall, f1 = train_single_model(model, X_train, y_train, X_test, y_test)
        name = model.__class__.__name__
        print("="*30)
        print(name)
        print("******  Results  *******")
        print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n")

        log_entry = pd.DataFrame([[name, acc*100]], columns=log_cols)
        log = log.append(log_entry)

        if acc > best_accuracy: 
            best_accuracy = acc 
            best_predict = preds
            best_model = deepcopy(model)
    
    print("="*30)
    return log, best_accuracy, best_predict, best_model