##########################
#   ARGS COMPROBATION    #
##########################

import argparse
parser = argparse.ArgumentParser(
    description="Template based of template/template.py")
parser.add_argument("base_work_dir",
                    help="Path to code base dir , i.e ~/DeepLearning_EM")
parser.add_argument("data_dir", help="Path to data base dir")
parser.add_argument("result_dir",
                    help="Path to where the resulting output of the job will "\
                    "be stored")
parser.add_argument("-id", "--job_id", "--id", help="Job identifier", 
                    default="unknown_job")
parser.add_argument("-rid","--run_id", "--rid", help="Run number of the same job", 
                    type=int, default=0)
parser.add_argument("-gpu","--gpu", dest="gpu_selected", 
                    help="GPU number according to 'nvidia-smi' command",
                    required=True)
args = parser.parse_args()


##########################
#        PREAMBLE        #
##########################

import os
import sys
sys.path.insert(0, args.base_work_dir)
sys.path.insert(0, os.path.join(args.base_work_dir, 'oztel_2017'))

# Working dir
os.chdir(args.base_work_dir)

# Limit the number of threads
from util import limit_threads, set_seed, create_plots, store_history,\
                 TimeHistory, threshold_plots, save_img
limit_threads()

# Try to generate the results as reproducible as possible
set_seed(42)

crops_made = False
job_identifier = args.job_id + '_' + str(args.run_id)


##########################
#        IMPORTS         #
##########################

import random
import numpy as np
import keras
import math
import time
import tensorflow as tf
from data_manipulation import load_data, crop_data, merge_data_without_overlap,\
                              crop_data_with_overlap, merge_data_with_overlap, \
                              check_binary_masks, load_data_from_dir
from data_generators import keras_da_generator, ImageDataGenerator,\
                            keras_gen_samples, calculate_z_filtering
from cnn_oztel import cnn_oztel_2017
from metrics import jaccard_index, jaccard_index_numpy, voc_calculation,\
                    DET_calculation
from itertools import chain
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from PIL import Image
from tqdm import tqdm
from smooth_tiled_predictions import predict_img_with_smooth_windowing, \
                                     predict_img_with_overlap
from skimage.segmentation import clear_border
from keras.utils.vis_utils import plot_model
from util import divide_images_on_classes
import shutil


############
#  CHECKS  #
############

print("Arguments: {}".format(args))
print("Python       : {}".format(sys.version.split('\n')[0]))
print("Numpy        : {}".format(np.__version__))
print("Keras        : {}".format(keras.__version__))
print("Tensorflow   : {}".format(tf.__version__))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_selected;


##########################                                                      
#  EXPERIMENT VARIABLES  #
##########################

### Dataset variables
# Main dataset data/mask paths
train_path = os.path.join(args.data_dir, 'train', 'x')
train_mask_path = os.path.join(args.data_dir, 'train', 'y')
test_path = os.path.join(args.data_dir, 'test', 'x')
test_mask_path = os.path.join(args.data_dir, 'test', 'y')
# Percentage of the training data used as validation                            
perc_used_as_val = 0.1
# Create the validation data with random images of the training data. If False
# the validation data will be the last portion of training images.
random_val_data = True


### Dataset shape
# Note: train and test dimensions must be the same when training the network and
# making the predictions. Be sure to take care of this if you are not going to
# use "crop_data()" with the arg force_shape, as this function resolves the 
# problem creating always crops of the same dimension
img_train_shape = [1024, 768, 1]
img_test_shape = [1024, 768, 1]


### Extra datasets variables
# Paths, shapes and discard values for the extra dataset used together with the
# main train dataset, provided by train_path and train_mask_path variables, to 
# train the network with. If the shape of the datasets differ the best option
# to normalize them is to make crops ("make_crops" variable)
extra_datasets_data_list = []
extra_datasets_mask_list = []
extra_datasets_data_dim_list = []
extra_datasets_discard = []
### Example of use:
# Path to the data:
# extra_datasets_data_list.append(os.path.join('kasthuri_pp', 'reshaped_fibsem', 'train', 'x'))
# Path to the mask: 
# extra_datasets_mask_list.append(os.path.join('kasthuri_pp', 'reshaped_fibsem', 'train', 'y'))
# Shape of the images:
# extra_datasets_data_dim_list.append([877, 967, 1])
# Discard value to apply in the dataset (see "Discard variables" for more details):
# extra_datasets_discard.append(0.05)                                             
#
# Number of crop to take form each dataset to train the network. If 0, the      
# variable will be ignored                                                      
num_crops_per_dataset = 0


### Crop variables
# Shape of the crops
crop_shape = (32, 32, 1)
# Flag to make crops on the train data
make_crops = False
# Flag to check the crops. Useful to ensure that the crops have been made 
# correctly. Note: if "discard_cropped_images" is True only the run that 
# prepare the discarded data will check the crops, as the future runs only load 
# the crops stored by this first run
check_crop = True 
# Instead of make the crops before the network training, this flag activates
# the option to extract a random crop of each train image during data 
# augmentation (with a crop shape defined by "crop_shape" variable). This flag
# is not compatible with "make_crops" variable
random_crops_in_DA = False 
# NEEDED CODE REFACTORING OF THIS SECTION
test_ov_crops = 8 # Only active with random_crops_in_DA
probability_map = False # Only active with random_crops_in_DA                       
w_foreground = 0.94 # Only active with probability_map
w_background = 0.06 # Only active with probability_map


### Discard variables
# Flag to activate the discards in the main train data. Only active when 
# "make_crops" variable is True
discard_cropped_images = False
# Percentage of pixels labeled with the foreground class necessary to not 
# discard the image 
d_percentage_value = 0.05
# Path where the train discarded data will be stored to be loaded by future runs 
# instead of make again the process
train_crop_discard_path = \
    os.path.join(args.result_dir, 'data_d'+str(d_percentage_value), 'train', 'x')
# Path where the train discarded masks will be stored                           
train_crop_discard_mask_path = \
    os.path.join(args.result_dir, 'data_d'+str(d_percentage_value), 'train', 'y')
# The discards are NOT done in the test data, but this will store the test data,
# which will be cropped, into the pointed path to be loaded by future runs      
# together with the train discarded data and masks                              
test_crop_discard_path = \
    os.path.join(args.result_dir, 'data_d'+str(d_percentage_value), 'test', 'x')
test_crop_discard_mask_path = \
    os.path.join(args.result_dir, 'data_d'+str(d_percentage_value), 'test', 'y')


### Normalization
# Flag to normalize the data dividing by the mean pixel value
normalize_data = False                                                          
# Force the normalization value to the given number instead of the mean pixel 
# value
norm_value_forced = -1                                                          


### Data augmentation (DA) variables
# Flag to decide which type of DA implementation will be used. Select False to 
# use Keras API provided DA, otherwise, a custom implementation will be used
custom_da = True
# Create samples of the DA made. Useful to check the output images made. 
# This option is available for both Keras and custom DA
aug_examples = True 
# Flag to shuffle the training data on every epoch 
#(Best options: Keras->False, Custom->True)
shuffle_train_data_each_epoch = custom_da
# Flag to shuffle the validation data on every epoch
# (Best option: False in both cases)
shuffle_val_data_each_epoch = False
# Make a bit of zoom in the images. Only available in Keras DA
keras_zoom = False 
# width_shift_range (more details in Keras ImageDataGenerator class). Only 
# available in Keras DA
w_shift_r = 0.0
# height_shift_range (more details in Keras ImageDataGenerator class). Only      
# available in Keras DA
h_shift_r = 0.0
# shear_range (more details in Keras ImageDataGenerator class). Only available 
# in Keras DA
shear_range = 0.0 
# Range to pick a brightness value from to apply in the images. Available for 
# both Keras and custom DA. Example of use: brightness_range = [1.0, 1.0]
brightness_range = None 
# Range to pick a median filter size value from to apply in the images. Option
# only available in custom DA
median_filter_size = [0, 0] 
# Range of rotation
rotation_range = 180
# To make rotation of 90º, -90º  and 180º. Only available in Custom DA
rotation90 = False
# Flag to make flips on the subvolumes. Available for both Keras and custom DA.
flips = False


### Extra train data generation
# Number of times to duplicate the train data. Useful when "random_crops_in_DA"
# is made, as more original train data can be cover
duplicate_train = 0
# Extra number of images to add to the train data. Applied after duplicate_train 
extra_train_data = 0


### Load previously generated model weigths
# Flag to activate the load of a previous training weigths instead of train 
# the network again
load_previous_weights = False
# ID of the previous experiment to load the weigths from 
previous_job_weights = args.job_id
# Flag to activate the fine tunning
fine_tunning = False
# ID of the previous weigths to load the weigths from to make the fine tunning 
fine_tunning_weigths = args.job_id
# Prefix of the files where the weights are stored/loaded from
weight_files_prefix = 'model.fibsem_'
# Name of the folder where weights files will be stored/loaded from. This folder 
# must be located inside the directory pointed by "args.base_work_dir" variable. 
# If there is no such directory, it will be created for the first time
h5_dir = os.path.join(args.result_dir, 'h5_files')


### Experiment main parameters
# Loss type, three options: "bce" or "w_bce_dice", which refers to binary cross 
# entropy (BCE) and BCE and Dice with with a weight term on each one (that must 
# sum 1) to calculate the total loss value. NOTE: "w_bce" is not implemented on 
# this template type: please use big_data_template.py instead.
loss_type = "bce"
# Batch size value
batch_size_value = 32
# Optimizer to use. Possible values: "sgd" or "adam"
optimizer = "adam"
# Learning rate used by the optimization method
learning_rate_value = 0.0001
# Number of epochs to train the network
epochs_value = 100
# Number of epochs to stop the training process after no improvement
patience = 20
# Flag to activate the creation of a chart showing the loss and metrics fixing 
# different binarization threshold values, from 0.1 to 1. Useful to check a 
# correct threshold value (normally 0.5)
make_threshold_plots = False
# Define time callback                                                          
time_callback = TimeHistory()
# If weights on data are going to be applied. To true when loss_type is 'w_bce' 
weights_on_data = True if loss_type == "w_bce" else False


### Network architecture specific parameters
# Number of channels in the first initial layer of the network
num_init_channels = 32 
# Flag to activate the Spatial Dropout instead of use the "normal" dropout layer
spatial_dropout = False
# Fixed value to make the dropout. Ignored if the value is zero
fixed_dropout_value = 0.0 
# Active flag if softmax is used as the last layer of the network
softmax_out = True

### Post-processing
# Flag to activate the post-processing (Smoooth and Z-filtering)
post_process = True


### DET metric variables
# More info of the metric at http://celltrackingchallenge.net/evaluation-methodology/ 
# and https://public.celltrackingchallenge.net/documents/Evaluation%20software.pdf
# NEEDED CODE REFACTORING OF THIS VARIABLE
det_eval_ge_path = os.path.join(args.result_dir, "..", 'cell_challenge_eval',
                                 'gen_' + job_identifier)
# Path where the evaluation of the metric will be done
det_eval_path = os.path.join(args.result_dir, "..", 'cell_challenge_eval', 
                             args.job_id, job_identifier)
# Path where the evaluation of the metric for the post processing methods will 
# be done
det_eval_post_path = os.path.join(args.result_dir, "..", 'cell_challenge_eval', 
                                  args.job_id, job_identifier + '_s')
# Path were the binaries of the DET metric is stored
det_bin = os.path.join(args.base_work_dir, 'cell_cha_eval' ,'Linux', 'DETMeasure')
# Number of digits used for encoding temporal indices of the DET metric
n_dig = "3"


### Paths of the results                                             
# Directory where predicted images of the segmentation will be stored
result_dir = os.path.join(args.result_dir, 'results', job_identifier)
# Directory where binarized predicted images will be stored
result_bin_dir = os.path.join(result_dir, 'binarized')
# Directory where predicted images will be stored
result_no_bin_dir = os.path.join(result_dir, 'no_binarized')
# Directory where binarized predicted images with 50% of overlap will be stored
result_bin_dir_50ov = os.path.join(result_dir, 'binarized_50ov')
# Directory where predicted images with 50% of overlap will be stored
result_no_bin_dir_50ov = os.path.join(result_dir, 'no_binarized_50ov')
# Folder where the smoothed images will be stored
smooth_dir = os.path.join(result_dir, 'smooth')
# Folder where the images with the z-filter applied will be stored
zfil_dir = os.path.join(result_dir, 'zfil')
# Folder where the images with smoothing and z-filter applied will be stored
smoo_zfil_dir = os.path.join(result_dir, 'smoo_zfil')
# Name of the folder where the charts of the loss and metrics values while 
# training the network will be shown. This folder will be created under the
# folder pointed by "args.base_work_dir" variable 
char_dir = os.path.join(result_dir, 'charts')
# Directory where weight maps will be stored                                    
loss_weight_dir = os.path.join(result_dir, 'loss_weights', args.job_id)
# Folder where smaples of DA will be stored
da_samples_dir = os.path.join(result_dir, 'aug')
# Folder where crop samples will be stored
check_crop_path = os.path.join(result_dir, 'check_crop')


#####################
#   SANITY CHECKS   #
#####################

print("#####################\n#   SANITY CHECKS   #\n#####################")

check_binary_masks(train_mask_path)
check_binary_masks(test_mask_path)
if extra_datasets_mask_list: 
    for i in range(len(extra_datasets_mask_list)):
        check_binary_masks(extra_datasets_mask_list[i])


##################################
#  OZTEL TRAIN DATA PREPARATION  #
##################################

print("##################################"
      "\n#  OZTEL TRAIN DATA PREPARATION  #"
      "\n##################################")

# Load train, val and test data. The validation data will be cropped, and test
# discarded as it will be loaded again after training data is prepared
X_train, Y_train, X_val,\
Y_val, X_test, Y_test,\
orig_test_shape, norm_value, _ = load_data(
    train_path, train_mask_path, test_path, test_mask_path,img_train_shape,
    img_test_shape, val_split=perc_used_as_val, shuffle_val=random_val_data, 
    make_crops=False)

X_val, Y_val, _ = crop_data(X_val, crop_shape, data_mask=Y_val)
del X_test, Y_test

# Divide the data into clases
p_train = os.path.join(args.result_dir, "data_train")
if os.path.exists(p_train) == False:
    X_train, Y_train, _ = crop_data(X_train, crop_shape, data_mask=Y_train)
    divide_images_on_classes(X_train, Y_train/255, p_train, th=0.8)    

# Balance the classes to have the same amount of samples
p_train_bal_x = os.path.join(p_train, "x", "balanced")
p_train_bal_y = os.path.join(p_train, "y", "balanced")
if os.path.exists(p_train_bal_x) == False:

    p_train_e_x = os.path.join(p_train, "x", "class1-extra")
    p_train_e_y = os.path.join(p_train, "y", "class1-extra")

    # Load mitochondria class labeled samples
    mito_data = load_data_from_dir(
        os.path.join(p_train, "x", "class1"), crop_shape)
    mito_mask_data = load_data_from_dir(
        os.path.join(p_train, "y", "class1"), crop_shape)

    # Calculate the number of samples to generate to balance the classes
    background_ids = len(next(os.walk(os.path.join(p_train, "x", "class0")))[2])   
    num_samples_extra = background_ids - mito_data.shape[0]

    # Create a generator 
    mito_gen_args = dict(
        X=mito_data, Y=mito_mask_data, batch_size=batch_size_value,
        dim=(crop_shape[0],crop_shape[1]), n_channels=1, shuffle=True, da=True,
        rotation90=True)
    mito_generator = ImageDataGenerator(**mito_gen_args)

    # Create the new samples
    extra_x, extra_y = mito_generator.get_transformed_samples(num_samples_extra)
    save_img(X=extra_x, data_dir=p_train_e_x, Y=extra_y, mask_dir=p_train_e_y, 
             prefix="e")
   
    # Copy the original samples and the extra ones into the same directory to 
    # read them later 
    print("Gathering all train samples into one folder . . .")
    os.makedirs(p_train_bal_x, exist_ok=True)
    os.makedirs(p_train_bal_y, exist_ok=True)
    for item in tqdm(os.listdir(os.path.join(p_train, "x", "class0"))):
        shutil.copy2(os.path.join(p_train, "x", "class0", item), p_train_bal_x)
    for item in tqdm(os.listdir(os.path.join(p_train, "y", "class0"))):
        shutil.copy2(os.path.join(p_train, "y", "class0", item), p_train_bal_y)
    for item in tqdm(os.listdir(os.path.join(p_train, "x", "class1"))):
        shutil.copy2(os.path.join(p_train, "x", "class1", item), p_train_bal_x)
    for item in tqdm(os.listdir(os.path.join(p_train, "y", "class1"))):
        shutil.copy2(os.path.join(p_train, "y", "class1", item), p_train_bal_y)
    for item in tqdm(os.listdir(p_train_e_x)):
        shutil.copy2(os.path.join(p_train_e_x, item), p_train_bal_x)
    for item in tqdm(os.listdir(p_train_e_y)):
        shutil.copy2(os.path.join(p_train_e_y, item), p_train_bal_y)


##########################
#       LOAD DATA        #
##########################

print("##################\n#    LOAD DATA   #\n##################\n")

# Load the data prepared
X_train, Y_train, X_test,\
Y_test, orig_test_shape, norm_value, _ = load_data(
    p_train_bal_x, p_train_bal_y, test_path, test_mask_path, 
    crop_shape, img_test_shape, create_val=False, make_crops=False)


##########################
#    DATA AUGMENTATION   #
##########################
 
print("##################\n#    DATA AUG    #\n##################\n")

# Custom Data Augmentation                                                  
data_gen_args = dict(
    X=X_train, Y=Y_train, batch_size=batch_size_value,     
    dim=(crop_shape[0],crop_shape[1]), n_channels=1,              
    shuffle=shuffle_train_data_each_epoch, da=True, rotation90=True,
    softmax_out=softmax_out)
    
data_gen_val_args = dict(
    X=X_val, Y=Y_val, batch_size=batch_size_value, 
    dim=(crop_shape[0],crop_shape[1]), n_channels=1,
    shuffle=shuffle_val_data_each_epoch, da=False, softmax_out=softmax_out)

data_gen_test_args = dict(
    X=X_test, Y=Y_test, batch_size=batch_size_value,
    dim=(img_test_shape[1], img_test_shape[0]), n_channels=1, shuffle=False, 
    da=False, softmax_out=softmax_out)

train_generator = ImageDataGenerator(**data_gen_args)                       
val_generator = ImageDataGenerator(**data_gen_val_args)                     
test_generator = ImageDataGenerator(**data_gen_test_args)                     
                                                                            
# Generate examples of data augmentation                                    
if aug_examples == True:                                                    
    train_generator.get_transformed_samples(
        10, save_to_dir=True, train=False, out_dir=da_samples_dir)

                                                                                
##########################
#    BUILD THE NETWORK   #
##########################

print("###################\n#  TRAIN PROCESS  #\n###################\n")

print("Creating the network . . .")
model = cnn_oztel_2017(crop_shape, lr=learning_rate_value, optimizer=optimizer)

# Check the network created
model.summary(line_length=150)
os.makedirs(char_dir, exist_ok=True)
model_name = os.path.join(char_dir, "model_plot_" + job_identifier + ".png")
plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

if load_previous_weights == False:
    earlystopper = EarlyStopping(patience=patience, verbose=1, 
                                 restore_best_weights=True)
    
    os.makedirs(h5_dir, exist_ok=True)
    checkpointer = ModelCheckpoint(
        os.path.join(h5_dir, weight_files_prefix + job_identifier + '.h5'), 
        verbose=1, save_best_only=True)
    
    if fine_tunning == True:                                                    
        h5_file=os.path.join(h5_dir, weight_files_prefix + fine_tunning_weigths 
                             + '_' + args.run_id + '.h5')     
        print("Fine-tunning: loading model weights from h5_file: {}"\
              .format(h5_file))
        model.load_weights(h5_file)                                             
   
    results = model.fit_generator(
        train_generator, validation_data=val_generator,
        validation_steps=math.ceil(len(X_val)/batch_size_value),
        steps_per_epoch=math.ceil(len(X_train)/batch_size_value),
        epochs=epochs_value, 
        callbacks=[earlystopper, checkpointer, time_callback])
else:
    h5_file=os.path.join(h5_dir, weight_files_prefix + previous_job_weights 
                         + '_' + str(args.run_id) + '.h5')
    print("Loading model weights from h5_file: {}".format(h5_file))
    model.load_weights(h5_file)


#####################
#     INFERENCE     #
#####################

print("##################\n#    INFERENCE   #\n##################\n")

if random_crops_in_DA == False:
    # Evaluate to obtain the loss value and the Jaccard index (per crop)
    print("Evaluating test data . . .")
    score = model.evaluate_generator(test_generator, verbose=1)
    jac_per_crop = score[1]

    # Predict on test
    print("Making the predictions on test data . . .")
    preds_test = model.predict_generator(test_generator, verbose=1)
    
    if softmax_out == True:
        # Decode predicted images into the original one
        decoded_pred_test = np.zeros(Y_test.shape)
        for i in range(preds_test.shape[0]):
            decoded_pred_test[i] = np.expand_dims(preds_test[i,...,1], -1)
        preds_test = decoded_pred_test

    # Reconstruct the data to the original shape
    if make_crops == True:
        h_num = int(orig_test_shape[1] / preds_test.shape[1]) \
                + (orig_test_shape[1] % preds_test.shape[1] > 0)
        v_num = int(orig_test_shape[2] / preds_test.shape[2]) \
                + (orig_test_shape[2] % preds_test.shape[2] > 0)
        
        #X_test = merge_data_without_overlap(
        #    X_test, math.ceil(X_test.shape[0]/(h_num*v_num)),
        #    out_shape=[h_num, v_num], grid=False)
        #Y_test = merge_data_without_overlap(
        #    Y_test, math.ceil(Y_test.shape[0]/(h_num*v_num)),
        #    out_shape=[h_num, v_num], grid=False)
        #print("The shape of the test data reconstructed is {}"
        #      .format(Y_test.shape))
        
        # To save the probabilities (no binarized)
        preds_test = merge_data_without_overlap(
            preds_test*255, math.ceil(preds_test.shape[0]/(h_num*v_num)),
            out_shape=[h_num, v_num], grid=False)
        preds_test = preds_test.astype(float)/255
        
    print("Saving predicted images . . .")
    save_img(Y=(preds_test > 0.5).astype(np.uint8), mask_dir=result_bin_dir, 
             prefix="test_out_bin")
    save_img(Y=preds_test, mask_dir=result_no_bin_dir, prefix="test_out_no_bin")

    # Metric calculation
    if make_threshold_plots == True:
        print("Calculate metrics with different thresholds . . .")
        score[1], voc, det = threshold_plots(
            preds_test, Y_test, orig_test_shape, score, det_eval_ge_path, 
            det_eval_path, det_bin, n_dig, args.job_id, job_identifier, char_dir)
    else:
        print("Calculate metrics . . .")
        # Per image without overlap
        score[1] = jaccard_index_numpy(Y_test, (preds_test > 0.5).astype(np.uint8))
        voc = voc_calculation(Y_test, (preds_test > 0.5).astype(np.uint8), score[1])
        det = DET_calculation(Y_test, (preds_test > 0.5).astype(np.uint8), 
                              det_eval_ge_path, det_eval_path, det_bin, n_dig, 
                              args.job_id)

        if make_crops == True:
            # Per image with 50% overlap
            Y_test_50ov = np.zeros(X_test.shape, dtype=(np.float32))
            for i in tqdm(range(0,len(X_test))):
                predictions_smooth = predict_img_with_overlap(
                    X_test[i,:,:,:],
                    window_size=crop_shape[0],
                    subdivisions=2,
                    nb_classes=1,
                    pred_func=(
                        lambda img_batch_subdiv: model.predict(img_batch_subdiv)
                    )
                )
                Y_test_50ov[i] = predictions_smooth
    
            print("Saving 50% overlap predicted images . . .")
            save_img(Y=(Y_test_50ov > 0.5).astype(np.float32), 
                     mask_dir=result_bin_dir_50ov, prefix="test_out_bin_50ov")
            save_img(Y=Y_test_50ov, mask_dir=result_no_bin_dir_50ov,
                     prefix="test_out_no_bin_50ov")
        
            print("Calculate metrics for 50% overlap images . . .")
            jac_per_img_50ov = jaccard_index_numpy(
                Y_test, (Y_test_50ov > 0.5).astype(np.float32))
            voc_per_img_50ov = voc_calculation(
                Y_test, (Y_test_50ov > 0.5).astype(np.float32), jac_per_img_50ov)
            det_per_img_50ov = DET_calculation(
                Y_test, (Y_test_50ov > 0.5).astype(np.float32), det_eval_ge_path, 
                det_eval_path, det_bin, n_dig, args.job_id)
        else:
            jac_per_img_50ov = -1
            voc_per_img_50ov = -1
            det_per_img_50ov = -1

    
####################
#  POST-PROCESING  #
####################

if post_process == True:

    print("##################\n# POST-PROCESING #\n##################\n")

    print("1) SMOOTH")

    Y_test_smooth = np.zeros(X_test.shape, dtype=(np.uint8))

    # Extract the number of digits to create the image names
    d = len(str(X_test.shape[0]))

    os.makedirs(smooth_dir, exist_ok=True)

    print("Smoothing crops . . .")
    for i in tqdm(range(0,len(X_test))):
        predictions_smooth = predict_img_with_smooth_windowing(
            X_test[i,:,:,:], window_size=crop_shape[0], subdivisions=2,  
            nb_classes=1, pred_func=(
                lambda img_batch_subdiv: model.predict(img_batch_subdiv)), 
            softmax=softmax_out)

        Y_test_smooth[i] = (predictions_smooth > 0.5).astype(np.uint8)

        im = Image.fromarray(predictions_smooth[:,:,0]*255)
        im = im.convert('L')
        im.save(os.path.join(smooth_dir,"test_out_smooth_" + str(i).zfill(d) 
                                        + ".png"))

    # Metrics (Jaccard + VOC + DET)
    print("Calculate metrics . . .")
    smooth_score = jaccard_index_numpy(Y_test, Y_test_smooth)
    smooth_voc = voc_calculation(Y_test, Y_test_smooth, smooth_score)
    smooth_det = DET_calculation(Y_test, Y_test_smooth, det_eval_ge_path,
                                 det_eval_post_path, det_bin, n_dig, args.job_id)

zfil_preds_test = None
smooth_zfil_preds_test = None
if post_process == True and not extra_datasets_data_list:
    print("2) Z-FILTERING")

    if random_crops_in_DA == False:
        print("Applying Z-filter . . .")
        zfil_preds_test = calculate_z_filtering((preds_test > 0.5).astype(np.uint8))
    else:
        if test_ov_crops > 1:
            print("Applying Z-filter . . .")
            zfil_preds_test = calculate_z_filtering(merged_preds_test)

    if zfil_preds_test is not None:
        print("Saving Z-filtered images . . .")
        save_img(Y=zfil_preds_test, mask_dir=zfil_dir, prefix="test_out_zfil")
 
        print("Calculate metrics for the Z-filtered data . . .")
        zfil_score = jaccard_index_numpy(Y_test, zfil_preds_test)
        zfil_voc = voc_calculation(Y_test, zfil_preds_test, zfil_score)
        zfil_det = DET_calculation(Y_test, zfil_preds_test, det_eval_ge_path,
                                   det_eval_post_path, det_bin, n_dig, 
                                   args.job_id)

    if Y_test_smooth is not None:
        print("Applying Z-filter to the smoothed data . . .")
        smooth_zfil_preds_test = calculate_z_filtering(Y_test_smooth)

        print("Saving smoothed + Z-filtered images . . .")
        save_img(Y=smooth_zfil_preds_test, mask_dir=smoo_zfil_dir, 
                 prefix="test_out_smoo_zfil")

        print("Calculate metrics for the smoothed + Z-filtered data . . .")
        smo_zfil_score = jaccard_index_numpy(Y_test, smooth_zfil_preds_test)
        smo_zfil_voc = voc_calculation(
            Y_test, smooth_zfil_preds_test, smo_zfil_score)
        smo_zfil_det = DET_calculation(
                Y_test, smooth_zfil_preds_test, det_eval_ge_path, 
                det_eval_post_path, det_bin, n_dig, args.job_id)

print("Finish post-processing") 


####################################
#  PRINT AND SAVE SCORES OBTAINED  #
####################################

if load_previous_weights == False:
    print("Epoch average time: {}".format(np.mean(time_callback.times)))
    print("Epoch number: {}".format(len(results.history['val_loss'])))
    print("Train time (s): {}".format(np.sum(time_callback.times)))
    print("Train loss: {}".format(np.min(results.history['loss'])))
    print("Train jaccard_index: {}"
          .format(np.max(results.history['jaccard_index_softmax'])))
    print("Validation loss: {}".format(np.min(results.history['val_loss'])))
    print("Validation jaccard_index: {}"
          .format(np.max(results.history['val_jaccard_index_softmax'])))

print("Test loss: {}".format(score[0]))
    
if random_crops_in_DA == False:    
    print("Test jaccard_index (per crop): {}".format(jac_per_crop))
    print("Test jaccard_index (per image without overlap): {}".format(score[1]))
    print("Test jaccard_index (per image with 50% overlap): {}"
          .format(jac_per_img_50ov))
    print("VOC (per image without overlap): {}".format(voc))
    print("VOC (per image with 50% overlap): {}".format(voc_per_img_50ov))
    print("DET (per image without overlap): {}".format(det))
    print("DET (per image with 50% overlap): {}".format(det_per_img_50ov))
else:
    print("Test overlapped (per crop) jaccard_index: {}".format(jac_per_crop))
    print("Test overlapped (per image) jaccard_index: {}".format(score[1]))
    if test_ov_crops > 1:
        print("VOC: {}".format(voc))
        print("DET: {}".format(det))
    
if load_previous_weights == False:
    smooth_score = -1 if 'smooth_score' not in globals() else smooth_score
    smooth_voc = -1 if 'smooth_voc' not in globals() else smooth_voc
    smooth_det = -1 if 'smooth_det' not in globals() else smooth_det
    zfil_score = -1 if 'zfil_score' not in globals() else zfil_score
    zfil_voc = -1 if 'zfil_voc' not in globals() else zfil_voc
    zfil_det = -1 if 'zfil_det' not in globals() else zfil_det
    smo_zfil_score = -1 if 'smo_zfil_score' not in globals() else smo_zfil_score
    smo_zfil_voc = -1 if 'smo_zfil_voc' not in globals() else smo_zfil_voc
    smo_zfil_det = -1 if 'smo_zfil_det' not in globals() else smo_zfil_det
    jac_per_crop = -1 if 'jac_per_crop' not in globals() else jac_per_crop

    store_history(
        results, jac_per_crop, score, jac_per_img_50ov, voc, voc_per_img_50ov, 
        det, det_per_img_50ov, time_callback, result_dir, job_identifier, 
        smooth_score, smooth_voc, smooth_det, zfil_score, zfil_voc, zfil_det, 
        smo_zfil_score, smo_zfil_voc, smo_zfil_det,
        metric="jaccard_index_softmax")

    create_plots(results, job_identifier, char_dir,
                 metric="jaccard_index_softmax")

if (post_process == True and make_crops == True) or (random_crops_in_DA == True):
    print("Post-process: SMOOTH - Test jaccard_index: {}".format(smooth_score))
    print("Post-process: SMOOTH - VOC: {}".format(smooth_voc))
    print("Post-process: SMOOTH - DET: {}".format(smooth_det))

if post_process == True and zfil_preds_test is not None:
    print("Post-process: Z-filtering - Test jaccard_index: {}".format(zfil_score))
    print("Post-process: Z-filtering - VOC: {}".format(zfil_voc))
    print("Post-process: Z-filtering - DET: {}".format(zfil_det))

if post_process == True and smooth_zfil_preds_test is not None:
    print("Post-process: SMOOTH + Z-filtering - Test jaccard_index: {}"
          .format(smo_zfil_score))
    print("Post-process: SMOOTH + Z-filtering - VOC: {}".format(smo_zfil_voc))
    print("Post-process: SMOOTH + Z-filtering - DET: {}".format(smo_zfil_det))

print("FINISHED JOB {} !!".format(job_identifier))