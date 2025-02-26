"""
OIDN structure:
data
`-- rt_train
    |-- scene1
    |   |-- view1_0001.alb.exr
    |   |-- view1_0001.hdr.exr
    |   |-- view1_0004.alb.exr
    |   |-- view1_0004.hdr.exr
    |   |-- view1_8192.alb.exr
    |   |-- view1_8192.hdr.exr
    |   |-- view2_0001.alb.exr
    |   |-- view2_0001.hdr.exr
    |   |-- view2_8192.alb.exr
    |   `-- view2_8192.hdr.exr
    |-- scene2_000008spp.alb.exr
    |-- scene2_000008spp.hdr.exr
    |-- scene2_000064spp.alb.exr
    |-- scene2_000064spp.hdr.exr
    |-- scene2_reference.alb.exr
    `-- scene2_reference.hdr.exr

We want to do something similiar. all things are 1 spp anyways.
we keep a list of transformation matrices somewhere in the folder (in transform.txt or something like that) or pre transform everything
actually pretransforming everything is better. we cannot do conv and then warp

so we assume that everything is pretransformed

we keep three folders:
- rt_train: training dataset
- rt_test: test dataset
- rt_fullres: full resolution images to dump to somewhere so we can see what the model looks like and is handmade (not outputted by this script)

we demodulate albedo instead of offloading that to the model

input features at each frame:
- reference
- demodulated color
- albedo
- normal
- world position (needed for warping)

since we need to calculate warped filtered, we cannot precalculate warped buffers
"""

import numpy as np

# We need this so OpenCV imports exr files
# in the real world we would be using minexr instead for faster file reads
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2
import minexr

import random

from multiprocessing import Pool

# todo, don't fix this
img_width = 1920
img_height = 1080
patch_size = 384

# iterate through all 8-tuple images
# idk what to do with the last 8 but it's really not important in the grand scheme of things
seq_len = 8

# we assume that the source dataset contains a bunch of regularly formatted images
# by that we mean a long string of images from the same frame sequence
src_dir = "data/src_dataset/"

# proportion of images we dump to the test directory (based on rng)
test_proportion = 0.05

def copy_patch(outdir, exr_cache, yoff, xoff, start_idx, idx, bufname, divtensor = None, albnorm = False):
    filename = str(start_idx + idx) + "-" + bufname + ".exr"
    filepath = src_dir + "/" + filename

    if(filepath in exr_cache):
        img = exr_cache[filepath]
    else:
        img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        exr_cache[filepath] = img

    patch = img[yoff:yoff+patch_size, xoff:xoff+patch_size, :]

    if(albnorm):
        patch[patch < 0.001] = 1.0

    if(divtensor is not None):
        patch = patch / divtensor

    # swap naming order because thats what Ive seen other datasets do
    outpath = outdir + "/" + bufname.lower() + "-" + str(idx) + ".exr"
    cv2.imwrite(outpath, patch, [cv2.IMWRITE_EXR_COMPRESSION, 0])

    return patch



def output_sample(outdir, exr_cache, start_idx, yoff, xoff):
    os.makedirs(outdir)
    for i in range(seq_len):
        alb = copy_patch(outdir, exr_cache, yoff, xoff, start_idx, i, "Albedo", albnorm=True)
        copy_patch(outdir, exr_cache, yoff, xoff, start_idx, i, "Color", alb)

        copy_patch(outdir, exr_cache, yoff, xoff, start_idx, i, "WorldNormal")
        copy_patch(outdir, exr_cache, yoff, xoff, start_idx, i, "Reference")

"""
Splits sequence into patches.
"""
def split_sequence(seq_idx):
    print(f"Processing full resolution sequence {seq_idx + 1}/{num_seq}")
    start_idx = seq_idx * seq_len

    num_x_patches = (img_width - 1) // patch_size + 1
    num_y_patches = (img_height - 1) // patch_size + 1

    exr_cache = {}
    for yp in range(num_y_patches):
        yoff = yp * patch_size

        if yoff + patch_size > img_height:
            yoff = img_height - patch_size

        for xp in range(num_x_patches):
            print(f"\tProcessing patch {xp + yp * num_x_patches + 1}/{num_x_patches * num_y_patches}")

            xoff = xp * patch_size
            if xoff + patch_size > img_width:
                xoff = img_width - patch_size

            patch_idx = seq_idx * num_x_patches * num_y_patches + xp + yp * num_x_patches
            outdir = "data/rt_" + ("train" if random.random() > test_proportion else "test") + "/patch_" + str(patch_idx)

            # we hand things off to our good friend abstraction
            output_sample(outdir, exr_cache, start_idx, yoff, xoff)

if __name__ == "__main__":
    num_frames=0
    while True:
        if os.path.exists(src_dir+ "/" + str(num_frames) + "-Reference.exr"):
            num_frames+=1
        else:
            break

    print(f"Found {num_frames} in {src_dir}.")
    print(f"Exactly {num_frames % seq_len} will be excluded from the final dataset because you are too lazy to figure out what to do with them")

    num_seq = num_frames // seq_len

    pool = Pool(8)
    pool.map(split_sequence, range(num_seq))


        