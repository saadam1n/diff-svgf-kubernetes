import numpy as np
import openexr_numpy as exr

# for file stuff idk I don't care anymore this hurts my head
import os
import sys
import shutil

import random

if __name__ == "__main__":
    tile_size = 256



    if(len(sys.argv) < 3):
        print("Usage: <path to folder of files> <path to place to output everything> <test prob>")
        exit(-1)

    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    test_prob = float(sys.argv[3])

    train_dir = dst_dir + "/rt_train"
    test_dir = dst_dir + "/rt_test"

    all_renders = os.listdir(src_dir)
    print(f"found the following files: {all_renders}")

    for render in all_renders:
        render_dir = render
        render_path = src_dir + "/" + render_dir

        print(f"Processing path {render_path}")

        if random.random() < test_prob:
            # we can't do a simple copy because we have to change the format of the motion vectors
            shutil.copytree(render_path, test_dir + "/" + render_dir)

    