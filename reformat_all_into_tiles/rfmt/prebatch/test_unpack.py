import torch

import os
import shutil

import openexr_numpy as exr

exr.set_default_channel_names(2, ["r", "g"])

file = "/tmp/prebatch_test/test_fullres_dir/batch-0.pt"

output_location = "/tmp/unpack/"

if os.path.exists(output_location):
    shutil.rmtree(output_location)

os.makedirs(output_location)

parts = torch.load(file, weights_only=True)

seq_in = parts["seq_in"]
seq_ref = parts["seq_ref"]

def output_image(folder, tensor, base, size, name):
    image = tensor[base:base+size].permute(1, 2, 0).numpy()

    exr.imwrite(os.path.join(folder, name), image)


def output_sample(folder, sample_in, sample_ref):
    os.makedirs(folder)

    for i in range(8):
        ibase = i * 11
        rbase = i * 3

        output_image(folder, sample_in, ibase + 0, 3, f"color{i}.exr")
        output_image(folder, sample_in, ibase + 3, 3, f"albedo{i}.exr")
        output_image(folder, sample_in, ibase + 6, 3, f"normal{i}.exr")
        output_image(folder, sample_in, ibase + 9, 2, f"motionvec{i}.exr")

        output_image(folder, sample_ref, rbase, 3, f"reference{i}.exr")



for i in range(1):
    output_sample(os.path.join(output_location, f"sample{i}"), seq_in[i], seq_ref[i])

