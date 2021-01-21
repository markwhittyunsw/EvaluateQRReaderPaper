# Author: Gareth N Hill and Mark Whitty
# Plant and Food Research New Zealand and UNSW Sydney

#!/usr/bin/env python3

#! module load pfr-python3/3.6.5

print('Here we go...')

import sys
import pyqrcode
import os, os.path  # For PATH etc.  https://docs.python.org/2/library/os.html
import sys  # For handling command line args
from glob import glob  # For Unix style finding pathnames matching a pattern (like regexp)
import cv2  # Image processing, OpenCV
import pyzbar.pyzbar as pyzbar  # QR code processing
from pyzbar.pyzbar import ZBarSymbol
import numpy as np
from numpy import array
import datetime as dt
from datetime import datetime
import re  # Regular expressions used for file type matching
import random  # For making random characters
import string  # For handling strings
import shutil  # For file copying
from fpdf import FPDF
from PIL import Image  # For image rotation
import math  # For radians / degrees conversions
from input.image_manipulations import *
from input.functions import *
import skimage
from skimage import measure
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

from input.image_manipulations import *
from input.functions import *

# GET ARGUMENTS -----------------------------------------------------------------------
args = sys.argv
print(str(args[1]))

# Set the maximum number of iterations
max_iter = 1000 
if len(args) > 1:
    try:
        max_iter = int(args[1]) 
    except:
        print('max_iter argument not an integer.')
        sys.exit()

# How often to save a CSV (number of images)
save_every = int(max_iter / 2)
if len(args) > 2:
    try:
        save_every = int(args[2]) 
    except:
        print('save_every argument not an integer.')

# SETUP FOLDERS -----------------------------------------------------------------------
if not os.path.exists('input'):
        os.makedirs('input')
if not os.path.exists('output'):
        os.makedirs('output')
        
run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
run_id = randomStringswithDigitsAndSymbols(1, 5)[0] # Necessary for simultaneous jobs
results_id = run_datetime + '_' + run_id

print(results_id)

file_output_path = os.path.join(os.sep,'output','projects','qr_metadata','run_results',('qr_eval_' + results_id))
print(file_output_path)
file_input_path = os.path.join('input','backgrounds')

if not os.path.exists(file_output_path):
        os.makedirs(file_output_path)
if not os.path.exists(file_input_path):
        os.makedirs(file_input_path)
        
# SETUP INPUT IMAGES -----

print(os.path.abspath(file_input_path))
input_files = [f for f in os.listdir(file_input_path) if
               re.search(r'.*\.(jpg|png|bmp|dib|jpe|jpeg|jp2|tif|tiff)$', f)]
input_files = list(map(lambda x: os.path.join(file_input_path, x), input_files))
num_input_files = len(input_files)

if num_input_files < 1:
    print("Warning: No image files in input directory: ", file_input_path)
    exit(0)

# SETUP VARIABLES ------------------------------------------------------------------
start_time = dt.datetime.now()
max_image_dimension = 2000  # Maximum image dimension, if greater than this will be resized before processing (output image is also resized)

print(str(num_input_files), "background files in", file_input_path, "directory")
bg_file_list = input_files

code_centre = (200, 100)  # Origin of the centre of the code (x, y) (x to the right, y down)


# ANALYSIS SETUP -------------------------------------------------------------------------------
results = pd.DataFrame()
image_count = 0
save_count = 1
max_images = int((max_iter - max_iter % len(bg_file_list)) / len(bg_file_list)) # Set maximum number of iterations per background image

# RUN ANALYSIS -------------------------------------------------------------------------------------
for i in range(max_images):
    #Randomise variables
    code_chars = np.random.randint(32,513)
    code_content = randomStringswithDigitsAndSymbols(1, code_chars - 5)[0] + run_id
    code_redundancy = np.random.choice(['L','M','H'])
    code_pixels = np.random.randint(25,201)
    rotation = np.random.randint(0,46)
    perspective = np.random.uniform(0.5,1.0)
    noise = np.random.uniform(0.00,0.10)
    shadow = [0]
    saturation = [0]
    bright_value = np.random.randint(128,256)
    dark_value = np.random.randint(0,256)
    # blur = np.random.uniform(0.0,0.025)
    compression = np.random.randint(2,101)

    # Generate QR code
    qr_array_binary = generate_QR_code(code_content, code_redundancy)

    # Add salt and pepper noise as well as set dark and bright values
    qr_array_255 = maw_add_sp_noise_and_set_dark_and_bright_values(
        qr_array_binary, noise, dark_value, bright_value)

    # Arbitrarily increase image size before further manipulations that may
    # degrade its quality
    qr_scale_factor = 30
    qr_array_scaled_up = maw_scale_up_with_alpha(qr_array_255, qr_scale_factor)

    # Add perspective effects
    qr_warped = maw_warp_perspective(qr_array_scaled_up, perspective)

    # Rotate QR code in image, handling background as alpha, so extra area can
    # be pasted over a background
    qr_array_rotated = maw_rotate_image(qr_warped, rotation)

    # Resize the QR code to be the desired size
    scale_factor = (code_pixels / len(qr_array_255)) / qr_scale_factor
    qr_array_rescaled = cv2.resize(qr_array_rotated, None, fx=scale_factor,
            fy=scale_factor, interpolation=cv2.INTER_AREA)
    

    # # Blur QR code
    # blur_kernel = int(qr_height * blur)
    # print(qr_height)
    # print(blur_kernel)
    # if blur_kernel > 0:
    #     qr_array_blurred = cv2.blur(qr_array_rescaled,(blur_kernel,blur_kernel))
    # else:
    #     qr_array_blurred = qr_array_rescaled

    qr_code = qr_array_rescaled

    # Write QR codes only
    qr_code_id = ("CC-" + code_content[:5] + "_CH-" + str("{:d}".format(code_chars)) + "_CR-" + code_redundancy
                    + "_CP-" + str("{:d}".format(code_pixels)) + "_CR-" + str("{:d}".format(rotation)) + "_P-" + str("{:3f}".format(perspective)) + "_N-" + str("{:3f}".format(noise))
                    + "_BV-" + str("{:d}".format(bright_value)) + "_DV-" + str("{:d}".format(dark_value))# + "_BL-" + str("{:3f}".format(blur))
                    + "_CO-" + str("{:d}".format(compression)))
    qr_code_img = Image.fromarray(qr_code)

    for bgfile in bg_file_list:

        image_count += 1

        # Save CSV whenever a certain number of images have been analysed
        if save_count == save_every:
            results.tail(save_every).to_csv(os.path.join(file_output_path,('temp_results_' + str(image_count) + '_' + results_id + ".csv")),index=False)
            results = pd.DataFrame() ## Reset DataFrame once saved
            save_count = 1
            print('image %i / %i (File saved)' % (image_count, max_iter))
        else:
            save_count += 1
            print('image %i / %i' % (image_count, max_iter))

        background = cv2.imread(bgfile)
        if background is None:
            print("Warning: image " + bgfile + " could not be read")
            continue
        bg_entropy = measure.shannon_entropy(background)
        bg_file_size =  os.path.getsize(bgfile)

        # Get qr code size and placement
        b_height, b_width = background.shape[:2]
        qr_height, qr_width = qr_code.shape[:2]
        code_centre = (int((b_width / 2) - (qr_width / 2)), int((b_height / 2) - (qr_height / 2))) 

        # Write QR code onto image
        qr_overlay = maw_overlay_image(background, qr_code, code_centre)

        # Make filename
        output_filename = ("B-" + os.path.basename(bgfile) + "_CC-" + code_content[:5] + "_CH-" + str("{:d}".format(code_chars)) + "_CR-" + code_redundancy
                           + "_CP-" + str("{:d}".format(code_pixels)) + "_CR-" + str("{:d}".format(rotation)) + "_P-" + str("{:3f}".format(perspective)) + "_N-" + str("{:3f}".format(noise))
                           + "_BV-" + str("{:d}".format(bright_value)) + "_DV-" + str("{:d}".format(dark_value))# + "_BL-" + str("{:3f}".format(blur))
                           + "_CO-" + str("{:d}".format(compression)))
        image_path = os.path.join('output','temp', (output_filename + '.jpg'))
        cv2.imwrite(image_path, qr_overlay, (cv2.IMWRITE_JPEG_QUALITY, compression))
        image_to_decode = cv2.imread(image_path)
        new_image_size = os.path.getsize(image_path)

        bg_entropy = measure.shannon_entropy(background)
        overlaid_entropy = measure.shannon_entropy(cv2.imread(image_path))
        qr_entropy = measure.shannon_entropy(qr_code)

        # Write variables to DataFrame

        results.loc[image_count,'image_id'] = run_id + '_' + str(image_count).zfill(len(str(max_images)))
        results.loc[image_count,'background'] = os.path.splitext(os.path.basename(bgfile))[0]
        results.loc[image_count,'background_entropy'] = bg_entropy
        results.loc[image_count,'background_file_size'] = bg_file_size
        results.loc[image_count,'code_content'] = code_content
        results.loc[image_count,'code_chars'] = len(code_content)
        results.loc[image_count,'code_redundancy'] = code_redundancy
        results.loc[image_count,'code_size_pixels'] = code_pixels 
        results.loc[image_count,'rotation_angle'] = rotation
        results.loc[image_count,'perspective'] = perspective
        results.loc[image_count,'noise'] = noise
        results.loc[image_count,'bright_value'] = bright_value
        results.loc[image_count,'dark_value'] = dark_value
        # results.loc[image_count,'blur'] = blur
        results.loc[image_count,'compression'] = compression 
        results.loc[image_count,'qr_code_id'] = qr_code_id
        results.loc[image_count,'image_entropy'] = overlaid_entropy
        results.loc[image_count,'image_file_size'] = new_image_size

        decodedObject = decode(image_to_decode, output_filename)

        if os.path.exists(image_path):
            os.remove(image_path)

        if decodedObject is None:
            results.loc[image_count,'status'] = 0 # Fail
            results.loc[image_count,'data_read'] = "-"

            continue
        else:
            qr_data = str(decodedObject.data)[2:-1]

            if qr_data == code_content:
                results.loc[image_count,'status'] = 2 # 1 = Success
                results.loc[image_count,'data_read'] = qr_data

            else:
                results.loc[image_count,'status'] = 1 # Read, but wrong code
                results.loc[image_count,'data_read'] = qr_data

# SAVE ALL RESULTS AND REMOVE TEMP FILES -----------------------------------------------------------------------
temp_files = glob(os.path.join(file_output_path,'temp_*'))

# Concatenate temp files
all_results = pd.concat(pd.read_csv(f).assign(file_id = run_id) for f in temp_files)
all_results.to_csv(os.path.join(file_output_path,('results_' + run_id + '.csv.gz')), compression='gzip', index=False)

# Remove temp files
for file in temp_files:
    os.remove(file)
