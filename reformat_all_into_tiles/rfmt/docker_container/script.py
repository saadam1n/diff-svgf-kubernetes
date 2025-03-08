import numpy as np
import openexr_numpy as exr

# for file stuff idk I don't care anymore this hurts my head
import os
import sys
import shutil

import random
from multiprocessing import Pool

"""
Stuff to process:
- Infinigen outputs
- Manually rendered outputs and BMFR
- Noisebase
"""

#random.seed(2252025809)

class FrameSequence:
    def __init__(self, path):
        self.path = path

        self.num_frames = 0
        while os.path.exists(self.path + f"reference{self.num_frames}.exr"):
            self.num_frames += 1

        if self.num_frames == 0:
            raise RuntimeError(f"Found no frames at path {self.path}")

        print(f"Found {self.num_frames} at path {self.path}")

        self.albedo = self.load_buffer("albedo")
        self.color = self.load_buffer("color")
        self.depth = self.load_buffer("depth")
        self.motionvec = np.zeros_like(self.color[:, :, :, :2])
        self.normal = self.load_buffer("normal")
        self.position = self.load_buffer("position")
        self.reference = self.load_buffer("reference")

        self.resx = self.reference.shape[2]
        self.resy = self.reference.shape[1]

        # adjust motionvec to be in terms of pixels so we don't have to do resolution-dependent scaling
        self.motionvec[:, :, :, 0] *= self.resx
        self.motionvec[:, :, :, 1] *= self.resy

    """
    Return dimensions: B x H x W X C
    """
    def load_buffer(self, name, channels=None):
        print(f"Loading buffer {name}")

        try:
            buffer_batch = self.load_buffer_batch(name, channels)
        except:
            print(f"Failed to load buffer {name} with channel config {channels}. Retrying with default configuration")
            buffer_batch = self.load_buffer_batch(name, None)
            buffer_batch = buffer_batch[:, :, :, 0:len(channels)]
            
        return buffer_batch

    def load_buffer_batch(self, name, channels):
        return np.stack([
            exr.imread(self.path + f"{name}{i}.exr", channel_names=channels) for i in range(self.num_frames)
        ])

    def slice_n_save(self, path, s0, s1, x0, y0, x1, y1, buffer, name):
        for s in range(s0, s1):
            exr.imwrite(path + f"{name}{s - s0}.exr", buffer[s, y0:y1, x0:x1])


    def save_sequence(self, path, s0, s1, x0, y0, x1, y1):
        os.makedirs(path, exist_ok=True)

        self.slice_n_save(path, s0, s1, x0, y0, x1, y1, self.albedo, "albedo")
        self.slice_n_save(path, s0, s1, x0, y0, x1, y1, self.color, "color")
        self.slice_n_save(path, s0, s1, x0, y0, x1, y1, self.depth, "depth")
        self.slice_n_save(path, s0, s1, x0, y0, x1, y1, self.motionvec, "motionvec")
        self.slice_n_save(path, s0, s1, x0, y0, x1, y1, self.normal, "normal")
        self.slice_n_save(path, s0, s1, x0, y0, x1, y1, self.position, "position")
        self.slice_n_save(path, s0, s1, x0, y0, x1, y1, self.reference, "reference")

if __name__ == "__main__":
    tile_size = 256
    seq_len = 8

    if(len(sys.argv) < 3):
        print("Usage: <path to folder of files> <path to place to output everything> <test prob>")
        exit(-1)

    src_dir = sys.argv[1] + "/"
    dst_dir = sys.argv[2] + "/"
    test_prob = float(sys.argv[3])

    train_dir = dst_dir + "rt_train/"
    test_dir = dst_dir + "rt_test/"
    test_fullres_dir = dst_dir + "test_fullres_dir/"

    exr.set_default_channel_names(2, ["r", "g"])

    all_renders = os.listdir(src_dir)
    print(f"Found the following rendered directories: {all_renders}")

    #for render in all_renders:
    def process_input(render):
        random.seed(os.getpid())

        render_dir = render
        render_path = src_dir + render_dir + "/"

        print(f"\n\n\nProcessing path {render_dir}")

        fs = FrameSequence(render_path)

        is_test_component = (random.random() < test_prob)
        
        dump_prefix = (test_dir if is_test_component else train_dir) + render_dir + "_"

        # save full resolution component if a test sequence
        if is_test_component:
            print("Current render has been picked to be in test set!")
        else:
            print("Current render has been picked to be in train set.")
    
        print("Be aware that this version of the tool automatically saves everything into the full resolution test sequence regardless.")
        fs.save_sequence(test_fullres_dir + render_dir + "/", 0, fs.num_frames, 0, 0, fs.resx, fs.resy)
        

        print("Dumping tiles!")
        next_dump_index = 0
        for i in range(0, fs.num_frames, seq_len):
            for yiter in range(0, fs.resy, tile_size):
                for xiter in range(0, fs.resx, tile_size):
                    ibase = i if i + seq_len <= fs.num_frames else fs.num_frames - seq_len
                    ybase = yiter if yiter + tile_size <= fs.resy else fs.resy - tile_size
                    xbase = xiter if xiter + tile_size <= fs.resx else fs.resx - tile_size

                    tile_dump_path = dump_prefix + str(next_dump_index) + "/"
                    print(f"Processing tile at base ({i},\t{ybase},\t{xbase}).\tDumping to {tile_dump_path}")

                    fs.save_sequence(tile_dump_path, ibase, ibase + seq_len, xbase, ybase, xbase + tile_size, ybase + tile_size)

                    next_dump_index += 1
    
    with Pool(16) as p:
        p.map(process_input, all_renders)

    print("Done!")