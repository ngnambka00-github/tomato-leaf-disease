import cv2 
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
import mahotas

from skimage.feature import hog
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.cluster import entropy as compute_entropy
from skimage.feature import local_binary_pattern
from hydra import initialize, compose
from omegaconf import OmegaConf

with initialize(config_path="../config/"):
    data_cfg = compose(config_name="hyper_parameter")
parameter_cfg = OmegaConf.create(data_cfg)

BINS = parameter_cfg.final_variable.hog_bins_feature

# anh truyen vao la anh mau
def hog_extraction(image):
    img = np.copy(image)
    # if len(img.shape) == 3:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    resized_image = cv2.resize(img, dsize=(64, 128))
    fd, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)

    return fd

# Anh truyen vao dang gray scale
def hog_extraction_custom(image, cell_size=8, block_size=2, bins=9): 
    img = np.copy(image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img = cv2.resize(src=img, dsize=(64, 128))

    h, w = img.shape # 128, 64
    
    # gradient
    xkernel = np.array([[-1, 0, 1]])
    ykernel = np.array([[-1], [0], [1]])
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel)
    
    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan(np.divide(dy, dx+0.00001)) # radian
    orientation = np.degrees(orientation) # -90 -> 90
    orientation += 90 # 0 -> 180
    
    num_cell_x = w // cell_size # 8
    num_cell_y = h // cell_size # 16
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins]) # 16 x 8 x 9
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
            mag = magnitude[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag) # 1-D vector, 9 elements
            hist_tensor[cy, cx, :] = hist
        pass
    pass
    
    # normalization
    redundant_cell = block_size-1
    feature_tensor = np.zeros([num_cell_y-redundant_cell, num_cell_x-redundant_cell, block_size*block_size*bins])
    for bx in range(num_cell_x-redundant_cell): # 7
        for by in range(num_cell_y-redundant_cell): # 15
            by_from = by
            by_to = by+block_size
            bx_from = bx
            bx_to = bx+block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten() # to 1-D array (vector)
            feature_tensor[by, bx, :] = v / LA.norm(v, 2)
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any(): # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v
    
    return feature_tensor.flatten() # 3780 features

# anh truyen vao la anh gray
def glcm_feature(gray_image):
    image = np.copy(gray_image)

    # 1. compute entropy
    # entropy = compute_entropy(image)

    # khỏi tạo ma trận mức xám GLCM
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)
    max_value = inds.max()+1
    # Các góc phi là 0, 45, 90, 135
    matrix_coocurrence = graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=True, symmetric=True)

     # 2. compute contract
    contrast = np.mean(graycoprops(matrix_coocurrence, 'contrast'))

    # 3. compute dissimilarity
    dissimilarity = np.mean(graycoprops(matrix_coocurrence, 'dissimilarity'))

    # 4. compute homogeneity
    homogeneity = np.mean(graycoprops(matrix_coocurrence, 'homogeneity'))

    # 5. compute energy
    energy = np.mean(graycoprops(matrix_coocurrence, 'energy'))

    # 6. compute correlation
    correlation = np.mean(graycoprops(matrix_coocurrence, 'correlation'))

    # 7. compute ASM
    asm = np.mean(graycoprops(matrix_coocurrence, 'ASM'))

    # return entropy, contrast, dissimilarity, homogeneity, energy, correlation, asm
    return contrast, dissimilarity, homogeneity, energy, correlation, asm

# Anh truyen vao dang gray scale
def hu_moments_feature(gray_image):
    gray = np.copy(gray_image)

    # Calculate Moments
    moments = cv2.moments(gray)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments).flatten()

    # Log scale hu moments
    for i in range(0,7):
        huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))    

    return huMoments

# anh truyen vao dang gray scale
def histogram_feature(gray_image, bins=10, range=[0, 256]):
    gray = np.copy(gray_image)

    hists, _ = np.histogram(gray, bins=bins, range=range)

    return hists

# feature dau ra la ma tran co kich thuoc bang anh dau vao
def LBP_features(gray_image):
    out_scikit = local_binary_pattern(image=gray_image, P=8, R=1, method='default')
    return out_scikit

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    gray = np.copy(image)
    if len(image.shape) == 3: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    gray = np.copy(image)
    
    if len(image.shape) == 3: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [BINS, BINS, BINS], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten() 