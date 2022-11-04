import os
import scipy
import numpy as np
from tqdm import tqdm
import pandas as pd
from skimage.segmentation import clear_border, find_boundaries
from skimage.io import imread
from scipy.ndimage.morphology import binary_dilation                                                    
from skimage.morphology import disk  
from skimage.measure import label

from utils.util import load_data_from_dir, load_3d_images_from_dir, save_npy_files, save_tif


def create_instance_channels(cfg, data_type='train'):
    """Create training and validation new data with appropiate channels based on ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS`` for instance
       segmentation.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

	   data_type: str, optional
		   Wheter to create training or validation instance channels.

       Returns
       -------
       filenames: List of str
           Image paths.
    """

    assert data_type in ['train', 'val']

    f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir
    tag = "TRAIN" if data_type == "train" else "VAL"
    Y, _, _, filenames = f_name(getattr(cfg.DATA, tag).MASK_PATH, return_filenames=True)
    print("Creating Y_{} channels . . .".format(data_type))
    if isinstance(Y, list):
        for i in tqdm(range(len(Y))):
            Y[i] = labels_into_bcd(Y[i], mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, save_dir=getattr(cfg.PATHS, tag+'_INSTANCE_CHANNELS_CHECK'),
                          fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE)
    else:
        Y = labels_into_bcd(Y, mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, save_dir=getattr(cfg.PATHS, tag+'_INSTANCE_CHANNELS_CHECK'),
                   fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE)
    save_npy_files(Y, data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR, filenames=filenames,
                   verbose=cfg.TEST.VERBOSE)
    X, _, _, filenames = f_name(getattr(cfg.DATA, tag).PATH, return_filenames=True)
    print("Creating X_{} channels . . .".format(data_type))
    save_npy_files(X, data_dir=getattr(cfg.DATA, tag).INSTANCE_CHANNELS_DIR, filenames=filenames,
                   verbose=cfg.TEST.VERBOSE)


def create_test_instance_channels(cfg):
    """Create test new data with appropiate channels based on ``PROBLEM.INSTANCE_SEG.DATA_CHANNELS`` for instance segmentation.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.
    """

    f_name = load_data_from_dir if cfg.PROBLEM.NDIM == '2D' else load_3d_images_from_dir

    if cfg.DATA.TEST.LOAD_GT and cfg.TEST.EVALUATE:
        Y_test, _, _, test_filenames = f_name(cfg.DATA.TEST.MASK_PATH, return_filenames=True)
        print("Creating Y_test channels . . .")
        if isinstance(Y_test, list):
            for i in tqdm(range(len(Y_test))):
                Y_test[i] = labels_into_bcd(Y_test[i], mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                                            fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE)
        else:
            Y_test = labels_into_bcd(Y_test, mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS, save_dir=cfg.PATHS.TEST_INSTANCE_CHANNELS_CHECK,
                                     fb_mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CONTOUR_MODE)
        save_npy_files(Y_test, data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_MASK_DIR, filenames=test_filenames,
                       verbose=cfg.TEST.VERBOSE)

    print("Creating X_test channels . . .")
    X_test, _, _, test_filenames = f_name(cfg.DATA.TEST.PATH, return_filenames=True)
    save_npy_files(X_test, data_dir=cfg.DATA.TEST.INSTANCE_CHANNELS_DIR, filenames=test_filenames,
                   verbose=cfg.TEST.VERBOSE)

def labels_into_bcd(data_mask, mode="BCD", fb_mode="outer", save_dir=None):
    """Create an array with 3 channels given semantic or instance segmentation data masks. These 3 channels are:
       semantic mask, contours and distance map.

       Parameters
       ----------
       data_mask : 5D Numpy array
           Data mask to create the new array from. It is expected to have just one channel. E.g. ``(10, 200, 1000, 1000, 1)``

       mode : str, optional
           Operation mode. Possible values: ``BC`` and ``BCD``.  ``BC`` corresponds to use binary segmentation+contour.
           ``BCD`` stands for binary segmentation+contour+distances.

       fb_mode : str, optional
          Mode of the find_boundaries function from ``scikit-image``. More info in:
          `find_boundaries() <https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries>`_.

       save_dir : str, optional
           Path to store samples of the created array just to debug it is correct.

       Returns
       -------
       new_mask : 5D Numpy array
           5D array with 3 channels instead of one. E.g. ``(10, 200, 1000, 1000, 3)``
    """

    assert data_mask.ndim in [5, 4]

    d_shape = 4 if data_mask.ndim == 5 else 3
    if mode in ['BCDv2', 'Dv2']:
        c_number = 4
    elif mode in ['BCD', 'BCM']:
        c_number = 3
    elif mode == 'BC':
        c_number = 2


    new_mask = np.zeros(data_mask.shape[:d_shape] + (c_number,), dtype=np.float32)

    for img in tqdm(range(data_mask.shape[0])):
        vol = data_mask[img,...,0].astype(np.int64)
        l = np.unique(vol)

        # If only have background -> skip
        if len(l) != 1:
            vol_dist = np.zeros(vol.shape)

            if mode in ["BCD", "BCDv2", "Dv2"]:
                # For each nucleus
                for i in tqdm(range(1,len(l)), leave=False):
                    obj = l[i]
                    distance = scipy.ndimage.distance_transform_edt(vol==obj)
                    vol_dist += distance

                # Foreground distance
                new_mask[img,...,2] = vol_dist.copy()

                # Background distance
                if mode in ["BCDv2", "Dv2"]:
                    # Background distance
                    vol_b_dist = np.invert(vol>0)
                    vol_b_dist= scipy.ndimage.distance_transform_edt(vol_b_dist)
                    vol_b_dist = np.max(vol_b_dist)-vol_b_dist
                    new_mask[img,...,3] = vol_b_dist.copy()

            # Semantic mask
            if mode != "Dv2":
                new_mask[img,...,0] = (vol>0).copy().astype(np.uint8)

            # Contour
            if mode in ["BCD", "BCDv2", "BC", "BCM", "Dv2"]:
                new_mask[img,...,1] = find_boundaries(vol, mode=fb_mode).astype(np.uint8)
                # Remove contours from segmentation maps
                new_mask[img,...,0][np.where(new_mask[img,...,1] == 1)] = 0
                if mode == "BCM":
                    new_mask[img,...,2] = (vol>0).astype(np.uint8)

    # Normalize and merge distance channels
    if mode in ["BCDv2", "Dv2"]:
        f_min = np.min(new_mask[...,2])
        f_max = np.max(new_mask[...,2])
        b_min = np.min(new_mask[...,3])
        b_max = np.max(new_mask[...,3])

        # Normalize foreground and background separately
        new_mask[...,2] = (new_mask[...,2]-f_min)/(f_max-f_min)
        new_mask[...,3] = (new_mask[...,3]-b_min)/(b_max-b_min)

        new_mask[...,2] = new_mask[...,3] - new_mask[...,2]
        # The intersection of the channels is the contour channel, so set it to the maximum value 1
        new_mask[...,2][new_mask[...,1]>0] = 1
        new_mask = new_mask[...,:3]
        if mode == "Dv2":
            new_mask = np.expand_dims(new_mask[...,-1], -1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = []
        if mode == "Dv2":
            suffix.append('_distance.tif')
        else:
            suffix.append('_semantic.tif')
        if mode in ["BC", "BCM", "BCD", "BCDv2"]:
            suffix.append('_contour.tif')
            if mode in ["BCD", "BCDv2"]:
                suffix.append('_distance.tif')
            elif mode == "BCM":
                suffix.append('_binary_mask.tif')

        for i in range(min(3,len(new_mask))):
            for j in range(len(suffix)):
                aux = np.transpose(new_mask[i,...,j],(2,0,1)) if data_mask.ndim == 5 else new_mask[i,...,j]
                aux = np.expand_dims(np.expand_dims(aux,-1),0)
                save_tif(aux, save_dir, filenames=['vol'+str(i)+suffix[j]], verbose=False)

    return new_mask

def create_detection_masks(cfg, data_type='train'):
    """Create detection masks based on CSV files.

       Parameters
       ----------
       cfg : YACS CN object
           Configuration.

	   data_type: str, optional
		   Wheter to create train, validation or test masks.
    """

    assert data_type in ['train', 'val', 'test']

    if data_type == "train":
        tag = "TRAIN"
    elif data_type == "train":
        tag = "VAL"
    else:
        tag = "TEST"
    img_dir = getattr(cfg.DATA, tag).PATH
    label_dir = getattr(cfg.DATA, tag).MASK_PATH
    out_dir = getattr(cfg.DATA, tag).DETECTION_MASK_DIR
    img_ids = sorted(next(os.walk(img_dir))[2])
    img_ext = '.'+img_ids[0].split('.')[-1]
    ids = sorted(next(os.walk(label_dir))[2])

    print("Creating {} detection masks . . .".format(data_type))
    for i in range(len(ids)):
        img_filename = ids[i].split('.')[0]+img_ext
        if not os.path.exists(os.path.join(out_dir, img_filename)):
            print("Attempting to create mask from CSV file: {}".format(os.path.join(label_dir, ids[i])))

            df = pd.read_csv(os.path.join(label_dir, ids[i]))  
            img = imread(os.path.join(img_dir, img_filename))
            
            # Adjust shape
            img = np.squeeze(img)
            if img.ndim == 3: 
                img = np.expand_dims(img, -1)

            # Discard first index column to not have error if it is not sorted 
            df = df.iloc[: , 1:]
            if len(df.columns) == 4:
                if not all(df.columns == ['axis-0', 'axis-1', 'axis-2', 'class']):
                    raise ValueError("CSV columns need to be ['axis-0', 'axis-1', 'axis-2', 'class']")
            elif len(df.columns) == 3:
                if not all(df.columns == ['axis-0', 'axis-1', 'axis-2']):
                    raise ValueError("CSV columns need to be ['axis-0', 'axis-1', 'axis-2']")
            else:
                raise ValueError("CSV file {} need to have 3 or 4 columns. Found {}"
                                 .format(os.path.join(label_dir, ids[i]), len(df.columns)))

            # Convert them to int in case they are floats
            df['axis-0'] = df['axis-0'].astype('int')
            df['axis-1'] = df['axis-1'].astype('int')
            df['axis-2'] = df['axis-2'].astype('int')
            
            # Obtain the points 
            z_axis_point = df['axis-0']                                                                       
            y_axis_point = df['axis-1']                                                                       
            x_axis_point = df['axis-2']
            
            # Class column present
            if len(df.columns) == 4:
                df['class'] = df['class'].astype('int')
                class_point = np.array(df['class'])            
            else:
                if cfg.MODEL.N_CLASSES > 1:
                    raise ValueError("MODEL.N_CLASSES > 1 but no class specified in CSV files (4th column must have class info)")
                class_point = [1] * len(z_axis_point)

            # Create masks
            print("Creating all points . . .")
            mask = np.zeros((img.shape[:-1]+(cfg.MODEL.N_CLASSES,)), dtype=np.uint8)
            for j in tqdm(range(len(z_axis_point)), total=len(z_axis_point), leave=False):
                z_coord = z_axis_point[j]
                y_coord = y_axis_point[j]
                x_coord = x_axis_point[j]
                c_point = class_point[j]-1

                if c_point+1 > mask.shape[-1]:
                    raise ValueError("Class {} detected while MODEL.N_CLASSES was set to {}. Please check it!"
                        .format(c_point+1, cfg.MODEL.N_CLASSES))

                # Paint the point
                if z_coord < mask.shape[0] and y_coord < mask.shape[1] and x_coord < mask.shape[2]:                                                                                                              
                    if 1 in mask[max(0,z_coord-1):min(mask.shape[0],z_coord+2),                                   
                                 max(0,y_coord-4):min(mask.shape[1],y_coord+5),                                   
                                 max(0,x_coord-4):min(mask.shape[2],x_coord+5), c_point]: 
                        print("WARNING: possible duplicated point in (3,9,9) neighborhood: coords {} , class {} "
                              "(point number {} in CSV)".format((z_coord,y_coord,x_coord), c_point, j))                                                                                                                                            
                                                                                                                            
                    mask[z_coord,y_coord,x_coord,c_point] = 1                                            
                    if y_coord+1 < mask.shape[1]: mask[z_coord,y_coord+1,x_coord,c_point] = 1       
                    if y_coord-1 > 0: mask[z_coord,y_coord-1,x_coord,c_point] = 1                   
                    if x_coord+1 < mask.shape[2]: mask[z_coord,y_coord,x_coord+1,c_point] = 1       
                    if x_coord-1 > 0: mask[z_coord,y_coord,x_coord-1,c_point] = 1                   
                else:  
                    print("WARNING: discarding point {} which seems to be out of shape: {}"
                          .format([z_coord,y_coord,x_coord], img.shape))                                                                                                     

            # Dilate the mask
            for k in range(mask.shape[0]): 
                for ch in range(mask.shape[-1]):                                                                                  
                    mask[k,...,ch] = binary_dilation(mask[k,...,ch], iterations=1,  structure=disk(3))                                                                                                                                                    
                
            if cfg.PROBLEM.DETECTION.CHECK_POINTS_CREATED:
                print("Check points created to see if some of them are very close that create a large label") 
                error_found = False
                for ch in range(mask.shape[-1]):
                    _, index, counts = np.unique(label(clear_border(mask[...,ch])), return_counts=True, return_index=True)                     
                    # 0 is background so valid element is 1. We will compare that value with the rest                                                                         
                    ref_value = counts[1]                                                                                           
                    for k in range(2,len(counts)):                                                                                  
                        if abs(ref_value - counts[k]) > 5:                                                                          
                            point = np.unravel_index(index[k], mask[...,ch].shape)  
                            print("WARNING: There is a point (coords {}) with size very different from "
                                  "the rest. Maybe that cell has several labels: please check it! Normally all point "
                                  "have {} pixels but this one has {}.".format(point, ref_value, counts[k])) 
                            error_found = True

                if error_found:
                    raise ValueError("Duplicate points have been found so please check them before continuing. "
                                     "If you consider that the points are valid simply disable "
                                     "'PROBLEM.DETECTION.CHECK_POINTS_CREATED' so this check is not done again!")

            save_tif(np.expand_dims(mask,0), out_dir, [img_filename])
        else:
            print("Mask file {} found for CSV file: {}".format(os.path.join(out_dir, img_filename), 
                os.path.join(label_dir, ids[i])))

def calculate_2D_volume_prob_map(Y, Y_path=None, w_foreground=0.94, w_background=0.06, save_dir=None):
    """Calculate the probability map of the given 2D data.

       Parameters
       ----------
       Y : 4D Numpy array
           Data to calculate the probability map from. E. g. ``(num_of_images, y, x, channel)``

       Y_path : str, optional
           Path to load the data from in case ``Y=None``.

       w_foreground : float, optional
           Weight of the foreground. This value plus ``w_background`` must be equal ``1``.

       w_background : float, optional
           Weight of the background. This value plus ``w_foreground`` must be equal ``1``.

       save_dir : str, optional
           Path to the file where the probability map will be stored.

       Raises
       ------
       ValueError
           if ``Y`` does not have 4 dimensions.

       ValueError
           if ``w_foreground + w_background > 1``.

       Returns
       -------
       Array : Str or 4D Numpy array
           Path where the probability map/s is/are stored if ``Y_path`` was given and there are images of different
           shapes. Otherwise, an array that represents the probability map of ``Y`` or all loaded data files from
           ``Y_path`` will be returned.
    """

    if Y is not None:
        if Y.ndim != 4:
            raise ValueError("'Y' must be a 4D Numpy array")

    if Y is None and Y_path is None:
        raise ValueError("'Y' or 'Y_path' need to be provided")

    if Y is not None:
        prob_map = np.copy(Y).astype(np.float32)
        l = prob_map.shape[0]
        channels = prob_map.shape[-1]
        v = np.max(prob_map)
    else:
        prob_map, _, _ = load_data_from_dir(Y_path)
        l = len(prob_map)
        channels = prob_map[0].shape[-1]
        v = np.max(prob_map[0])

    if isinstance(prob_map, list):
        first_shape = prob_map[0][0].shape
    else:
        first_shape = prob_map[0].shape

    print("Connstructing the probability map . . .")
    maps = []
    diff_shape = False
    for i in tqdm(range(l)):
        if isinstance(prob_map, list):
            _map = prob_map[i][0].copy().astype(np.float32)
        else:
            _map = prob_map[i].copy().astype(np.float32)

        for k in range(channels):
            # Remove artifacts connected to image border
            _map[:,:,k] = clear_border(_map[:,:,k])

            foreground_pixels = (_map[:,:,k] == v).sum()
            background_pixels = (_map[:,:,k] == 0).sum()

            if foreground_pixels == 0:
                _map[:,:,k][np.where(_map[:,:,k] == v)] = 0
            else:
                _map[:,:,k][np.where(_map[:,:,k] == v)] = w_foreground/foreground_pixels
            if background_pixels == 0:
                _map[:,:,k][np.where(_map[:,:,k] == 0)] = 0
            else:
                _map[:,:,k][np.where(_map[:,:,k] == 0)] = w_background/background_pixels

            # Necessary to get all probs sum 1
            s = _map[:,:,k].sum()
            if s == 0:
                t = 1
                for x in _map[:,:,k].shape: t *=x
                _map[:,:,k].fill(1/t)
            else:
                _map[:,:,k] = _map[:,:,k]/_map[:,:,k].sum()

        if first_shape != _map.shape: diff_shape = True
        maps.append(_map)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if not diff_shape:
            for i in range(len(maps)):
                maps[i] = np.expand_dims(maps[i], 0)
            maps = np.concatenate(maps)
            print("Saving the probability map in {}".format(save_dir))
            np.save(os.path.join(save_dir, 'prob_map.npy'), maps)
            return maps
        else:
            print("As the files loaded have different shapes, the probability map for each one will be stored"
                  " separately in {}".format(save_dir))
            d = len(str(l))
            for i in range(l):
                f = os.path.join(save_dir, 'prob_map'+str(i).zfill(d)+'.npy')
                np.save(f, maps[i])
            return save_dir

def calculate_3D_volume_prob_map(Y, Y_path=None, w_foreground=0.94, w_background=0.06, save_dir=None):
    """Calculate the probability map of the given 3D data.

       Parameters
       ----------
       Y : 5D Numpy array
           Data to calculate the probability map from. E. g. ``(num_subvolumes, z, y, x, channel)``

       Y_path : str, optional
           Path to load the data from in case ``Y=None``.

       w_foreground : float, optional
           Weight of the foreground. This value plus ``w_background`` must be equal ``1``.

       w_background : float, optional
           Weight of the background. This value plus ``w_foreground`` must be equal ``1``.

       save_dir : str, optional
           Path to the directory where the probability map will be stored.

       Returns
       -------
       Array : Str or 5D Numpy array
           Path where the probability map/s is/are stored if ``Y_path`` was given and there are images of different
           shapes. Otherwise, an array that represents the probability map of ``Y`` or all loaded data files from
           ``Y_path`` will be returned.

       Raises
       ------
       ValueError
           if ``Y`` does not have 5 dimensions.
       ValueError
           if ``w_foreground + w_background > 1``.
    """

    if Y is not None:
        if Y.ndim != 5:
            raise ValueError("'Y' must be a 5D Numpy array")

    if Y is None and Y_path is None:
        raise ValueError("'Y' or 'Y_path' need to be provided")

    if Y is not None:
        prob_map = np.copy(Y).astype(np.float32)
        l = prob_map.shape[0]
        channels = prob_map.shape[-1]
        v = np.max(prob_map)
    else:
        prob_map, _, _ = load_3d_images_from_dir(Y_path)
        l = len(prob_map)
        channels = prob_map[0].shape[-1]
        v = np.max(prob_map[0])

    if isinstance(prob_map, list):
        first_shape = prob_map[0][0].shape
    else:
        first_shape = prob_map[0].shape

    print("Constructing the probability map . . .")
    maps = []
    diff_shape = False
    for i in range(l):
        if isinstance(prob_map, list):
            _map = prob_map[i][0].copy().astype(np.float64)
        else:
            _map = prob_map[i].copy().astype(np.float64)

        for k in range(channels):
            for j in range(_map.shape[0]):
                # Remove artifacts connected to image border
                _map[j,:,:,k] = clear_border(_map[j,:,:,k])
            foreground_pixels = (_map[:,:,:,k] == v).sum()
            background_pixels = (_map[:,:,:,k] == 0).sum()

            if foreground_pixels == 0:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == v)] = 0
            else:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == v)] = w_foreground/foreground_pixels
            if background_pixels == 0:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == 0)] = 0
            else:
                _map[:,:,:,k][np.where(_map[:,:,:,k] == 0)] = w_background/background_pixels

            # Necessary to get all probs sum 1
            s = _map[:,:,:,k].sum()
            if s == 0:
                t = 1
                for x in _map[:,:,:,k].shape: t *=x
                _map[:,:,:,k].fill(1/t)
            else:
                _map[:,:,:,k] = _map[:,:,:,k]/_map[:,:,:,k].sum()

        if first_shape != _map.shape: diff_shape = True
        maps.append(_map)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if not diff_shape:
            for i in range(len(maps)):
                maps[i] = np.expand_dims(maps[i], 0)
            maps = np.concatenate(maps)
            print("Saving the probability map in {}".format(save_dir))
            np.save(os.path.join(save_dir, 'prob_map.npy'), maps)
            return maps
        else:
            print("As the files loaded have different shapes, the probability map for each one will be stored "
                  "separately in {}".format(save_dir))
            d = len(str(l))
            for i in range(l):
                f = os.path.join(save_dir, 'prob_map'+str(i).zfill(d)+'.npy')
                np.save(f, maps[i])
            return save_dir

def norm_range01(x):
    norm_steps = {}
    if x.dtype == np.uint8:
        x = x/255
        norm_steps['div_255'] = 1
    elif x.dtype == np.uint16:
        if np.max(x) > 255:
            x = reduce_dtype(x, 0, 65535, out_min=0, out_max=1, out_type=np.float32)
            norm_steps['reduced_uint16'] = 1
    x = x.astype(np.float32)
    return x, norm_steps

def reduce_dtype(x, x_min, x_max, out_min=0, out_max=1, out_type=np.float32):
    return ((np.array((x-x_min)/(x_max-x_min))*(out_max-out_min))+out_min).astype(out_type)

def normalize(data, means, stds):
    return (data - means) / stds

def denormalize(data, means, stds):
    return (data * stds) + means