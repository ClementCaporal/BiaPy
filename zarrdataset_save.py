from itertools import repeat
import multiprocessing as mp
import time
import numpy as np
import zarr
import zarrdataset as zds
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

#####################
# ZARR DATASET LOADER

filename = r"H:\221024-PB_P_21_Emx1Cre_1921_6color\data\221024-PB_P_21_Emx1Cre_1921_6color res level 0.ome.zarr"
filename = r"H:\P_00\data_2\fused ChroMS_221107-PB_P_00_Emx1Cre_1433_C06_6color-0004.zarr"

patch_size = dict(Z=17, Y=128, X=128)
overlap = dict(Z=4, Y=30, X=30)

patch_sampler = zds.OverlapGridPatchSampler(patch_size=patch_size, overlap=overlap)
# patch_sampler = zds.PatchSampler(patch_size=patch_size)
axes_sample = "CZYX"
my_datasets = zds.ZarrDataset(
    [
    zds.ImagesDatasetSpecs(
        filenames=filename,
        source_axes="TCZYX",
        axes=axes_sample,
        data_group="0",
    )
    ],
    patch_sampler=patch_sampler,
    return_positions=True,
    return_worker_id=True
)

import torchvision

img_preprocessing = torchvision.transforms.Compose([
    zds.ToDtype(dtype=np.float32),
    # torchvision.transforms.Lambda(lambda x: self.norm_X(np.array(x))),
])

my_datasets.add_transform("images", img_preprocessing)

my_dataloader = DataLoader(my_datasets,
            num_workers=10,
            worker_init_fn=zds.zarrdataset_worker_init_fn,
            batch_size=1,
            pin_memory=False,
            )


#####################
# ZARR WRITER

def writer(zarr_array, obj, i=0):
    #print("writer")
    wid, patch_coords, img = obj
    patch_coords = patch_coords.numpy()[0]
    zarr_array[
        :,
        :,
        patch_coords[1, 0]:patch_coords[1, 1]-4,
        patch_coords[2, 0]:patch_coords[2, 1]-30,
        patch_coords[3, 0]:patch_coords[3, 1]-30,
        ] = img.numpy()[:,:,:-4,:-30,:-30]


#####################
# MAIN

if __name__ == "__main__":
    zarr_root_path = 'test.zarr'

    print("Creating zarr file...")

    root = zarr.open_group(zarr_root_path, mode='a')

    dataset = root.create_dataset(
        shape=(1, 3, 82, 10752, 9728),
        chunks=(1, 3, 13, 98, 98),
        name='test',
        overwrite=True,
        synchronizer=zarr.ProcessSynchronizer('.lock'),
    )

    # dataset[:] = np.random.randn(*dataset.shape)

    iterable = zip(
        repeat(dataset),
        my_dataloader,
        #range(100),
    )
    print("Starting pool...")

    
    with mp.get_context('spawn').Pool(10) as pool:
        for dataset, obj in tqdm(iterable):
            pool.apply(writer, (dataset, obj))

    #for dataset, obj, i in tqdm(iterable):
    #    writer(dataset, obj)