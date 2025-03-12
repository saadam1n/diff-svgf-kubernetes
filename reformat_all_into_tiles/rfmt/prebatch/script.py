import torch
import torch.multiprocessing as multiprocessing
import numpy as np
import openexr_numpy as exr

import sys
import os
import shutil
import tarfile
import random
import hashlib
import time

download_queue = multiprocessing.Queue()
read_queue = multiprocessing.Queue()

def download_tarball(file):
    local_path = os.path.join(os.environ["DOWNLOAD_CACHE"], f"local-{hashlib.sha256(file.encode('utf-8')).hexdigest()}/")
    tarball_path = os.path.join(local_path, "download.tgz")

    os.mkdir(local_path)
    shutil.copyfile(file, tarball_path)

    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(local_path)
    except:
        shutil.rmtree(local_path)

        download_queue.put("FAILURE")
        return

    download_queue.put(local_path)

def load_exr_as_tensor(path):
    image = torch.from_numpy(exr.imread(path)).permute(2, 0, 1)

    read_queue.put((path, image))

def cat_and_view(read_buffers, download_paths, seq_len, buffers):
    cat_list = [
        read_buffers[os.path.join(download_path, f"{buffer}{i}.exr")]
        for download_path in download_paths
        for i in range(seq_len)
        for buffer in buffers 
    ]

    combined = torch.cat(cat_list, dim=0)

    batch = combined.view(len(download_paths), -1, combined.shape[1], combined.shape[2])

    return batch



def prebatch_tensors(dst, samples, num_batches, batch_size, seq_len, buffers):
    pool = multiprocessing.Pool()

    buffers_with_ref = buffers + ["reference"]

    for i in range(num_batches):
        batch_path = os.path.join(dst, f"batch-{i}.pt")
        print(f"Prebatching batch {batch_path} with contents {samples[i] if batch_size == 1 else '(multiple tensors in this batch)'}")

        # enqueue download
        pool.map(download_tarball, samples[i * batch_size:(i + 1) * batch_size])
            
        # wait for batch_size downloads to finish
        bad_batch = False
        download_paths = []
        for j in range(batch_size):            
            download_path = download_queue.get()
            if download_path == "FAILURE":
                print(f"\tCorrupted data in batch {batch_path}. This script will proceed normally to flush buffers but will not save this batch.")
                bad_batch = True
                continue

            download_paths.append(download_path)

            read_args = [
                os.path.join(download_path, f"{buffer}{i}.exr")
                for i in range(seq_len)
                for buffer in buffers_with_ref 
            ]

            pool.map(load_exr_as_tensor, read_args)

            print(f"\tFinished downloading and extracting {j + 1}/{batch_size} items")

        # get all reads and put them in array
        read_buffers = dict()

        num_images = len(download_paths) * seq_len * len(buffers_with_ref)
        for j in range(num_images):
            path, image = read_queue.get()

            read_buffers[path] = image

            if (j + 1) % 100 == 0:
                print(f"\tFinished reading {j + 1}/{num_images} images to memory.")

        for download_path in download_paths:
            shutil.rmtree(download_path)

        if bad_batch:
            continue

        print(f"\tConcatenating...")
        seq_in = cat_and_view(read_buffers, download_paths, seq_len, buffers)
        seq_ref = cat_and_view(read_buffers, download_paths, seq_len, ["reference"])

        combined_data = {
            "seq_in": seq_in,
            "seq_ref": seq_ref
        }

        torch.save(combined_data, batch_path)

        print("\tDone with this batch!")

    pool.close()
    pool.terminate()
    pool.join()


if __name__ == "__main__":
    exr.set_default_channel_names(2, ["r", "g"])

    if(len(sys.argv) < 3):
        print("Usage: <path to folder of files> <path to place to output everything> <sequence length> <batch size> <buffers>")
        exit(-1)

    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    seq_len = int(sys.argv[3])
    batch_size = int(sys.argv[4])

    buffers = [sys.argv[i] for i in range(5, len(sys.argv))]

    os.makedirs(dst_dir, exist_ok=True)

    print(f"Received buffers {buffers}")
    with open(os.path.join(dst_dir, "format.txt"), "w") as f:
        for buffer in buffers:
            f.write(f"{buffer}\n") 

    test_fullres_dir = "test_fullres_dir"
    suffixes = ["rt_train", "rt_test", test_fullres_dir]
    for suffix in suffixes:
        actual_src_dir = os.path.join(src_dir, suffix)
        actual_dst_dir = os.path.join(dst_dir, suffix)

        os.makedirs(actual_dst_dir, exist_ok=True)

        files = [os.path.join(actual_src_dir, file) for file in os.listdir(actual_src_dir)]
        random.shuffle(files)

        print(f"Found {len(files)} items in {actual_src_dir}")

        if suffix == test_fullres_dir:
            batch_size = 1

        num_batches = len(files) // batch_size

        selected_samples = [files[i] for i in range(0, num_batches * batch_size) ]

        prebatch_tensors(actual_dst_dir, selected_samples, num_batches, batch_size, seq_len, buffers)
        




    