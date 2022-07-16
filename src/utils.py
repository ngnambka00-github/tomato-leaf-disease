import os
import glob
import h5py
import cv2
import time
import joblib
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from hydra import initialize, compose
from omegaconf import OmegaConf
from imblearn.over_sampling import SMOTE

from image_processing import img_segmentation, sobel_edge_detection_2
from feature_extraction import fd_haralick, fd_hu_moments, fd_histogram

# Global data config
with initialize(config_path="../config/"):
    data_cfg = compose(config_name="hyper_parameter")
parameter_cfg = OmegaConf.create(data_cfg)

RANDOM_SEED = parameter_cfg.final_variable.seed
TEST_SIZE_SPLIT = parameter_cfg.final_variable.test_size_split

# export image to features
def extract_feature_to_file(data_path, feature_path, label_path):
    # get the training labels
    train_labels = os.listdir(data_path)

    # sort the training labels
    train_labels.sort()

    # empty lists to hold feature vectors and labels
    features = []
    labels = []

    # loop over the training data sub-folders
    for training_name in train_labels:

        file_names = glob.glob(f"{os.path.join(data_path, training_name)}/**")

        # loop over the images in each sub-folder
        for i, file in tqdm(enumerate(file_names), desc=f"[STATUS] Folder: {training_name}"):
            # read the image
            image = cv2.imread(file)

            # Running Function Bit By Bit
            RGB_BGR = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            BGR_HSV = cv2.cvtColor(RGB_BGR, cv2.COLOR_RGB2HSV)
            IMG_SEGMENT = img_segmentation(RGB_BGR,BGR_HSV)
            
            # convert to gray image
            IMG_GRAY = cv2.cvtColor(IMG_SEGMENT, cv2.COLOR_RGB2GRAY)

            # sharpen image
            SOBEL_IMG = sobel_edge_detection_2(IMG_GRAY)

            # morphology image
            kernel = np.ones((3,3),np.uint8)
            MORPHOLOGY_IMG = cv2.morphologyEx(SOBEL_IMG, cv2.MORPH_OPEN, kernel)
            
            # Feature extraction
            fv_hu_moments = fd_hu_moments(MORPHOLOGY_IMG)
            fv_haralick   = fd_haralick(MORPHOLOGY_IMG)
            fv_histogram  = fd_histogram(IMG_SEGMENT)
            
            # Hoang: Hu moment 
            # Khanh: Haralick
            # Trinh: Histogram Feature Extraction

            # Concatenate 
            new_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
            
            # update the list of labels and feature vectors
            labels.append(training_name)
            features.append(new_feature)

    print("[STATUS] completed Feature Extraction Phase...")

    # encode the target labels
    le = LabelEncoder()
    target = le.fit_transform(labels)

    # save to file
    h5f_data = h5py.File(feature_path, 'w')
    h5f_data.create_dataset('dataset_1', data=features)

    h5f_label = h5py.File(label_path, 'w')
    h5f_label.create_dataset('dataset_1', data=target)

    h5f_data.close()
    h5f_label.close()

    # tra ve feature, label encoded, class name
    return features, target, le.classes_

# EDA (Exploration data analysiz)
# split dataset to X_train, y_train, X_test, y_test
def split_data(feature_path, label_path): 
    # read from file
    h5f_data  = h5py.File(feature_path, 'r')
    h5f_label = h5py.File(label_path, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string   = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    # normailize
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)

    # split data
    (X_train, X_test, y_train, y_test) = train_test_split(rescaled_features, global_labels, test_size=TEST_SIZE_SPLIT, random_state=RANDOM_SEED)

    # Oversampling -> Process Imbalance Data
    sm = SMOTE()
    X_train_os, y_train_os = sm.fit_resample(X_train, y_train)

    h5f_data.close()
    h5f_label.close()

    return (X_train_os, y_train_os), (X_test, y_test)


# Train single model
def train_single_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predicts = model.predict(X_test)

    # compute accuracy
    accuracy = accuracy_score(y_test, predicts)

    # compute precision
    precision = precision_score(y_test, predicts, average='macro')

    # compute recall
    recall = recall_score(y_test, predicts, average='macro')

    # compute f1_score
    f1 = f1_score(y_test, predicts, average='macro')

    return predicts, accuracy, precision, recall, f1


# Train nhieu model va chon ra model co do chinh xac cao nhat
def train_test_model_classification(models, X_train, y_train, X_test, y_test, log_path=None, best_model_path=None):
    print("TRAINING PROCESSING")

    log_cols=["Classifier", "Accuracy", "Precision", "Recall", "F1_Score", "Trainning_Time"]
    log = pd.DataFrame(columns=log_cols)

    best_accuracy = 0
    best_model = None
    best_predict = None
    for model_name, model in models: 

        time_start = time.time()
        preds, acc, precision, recall, f1 = train_single_model(model, X_train, y_train, X_test, y_test)
        time_end = time.time()
        time_execute = time_end - time_start

        name_from_class = model.__class__.__name__

        print("="*30)
        print(model_name)
        print("*******  Results  ********")
        print(f"Acc: {acc:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
        print(f"Training time: {time_execute:.4f}\n")

        log_entry = pd.DataFrame([[model_name, acc*100, precision*100, recall*100, f1*100, time_execute]], columns=log_cols)
        log = log.append(log_entry, ignore_index=True)

        if acc > best_accuracy: 
            best_accuracy = acc 
            best_predict = preds
            best_model = deepcopy(model)
    
    print("="*30)

    # save log
    if log_path is not None: 
        log.to_csv(log_path, index=False)
    
    # save best model with highest accuracy
    if best_model_path is not None: 
        joblib.dump(best_model, best_model_path)

        # load the model from disk
        # loaded_model = joblib.load(filename)
        # result = loaded_model.score(X_test, Y_test)
        # print(result)

    return log, best_accuracy, best_predict, best_model