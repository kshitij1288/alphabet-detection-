import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time



#Fetching the data
X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

roi = gray[upper_left[1]:bottom_right[1],
upper_left[0]:bottom_right(0)]

image_bw = im.pil.convert('L')
image_bw_resized = image_bw.resize({28,28}, Image.ANATIALIAS)
#invert camera
image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
pixel_filter = 20
#converting to scalar quantity
min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)

image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
max_pixel = np.max(image_bw_resized_inverted)

image_bw_resized_inverted_scaled = np.array(image_bw_resized_inverted_scaled)/max_pixel
#creating a test sample and marking prediction
test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1.784)
test_pred = clf.predict(test_sample)
print("predicted class is:", test_pred)