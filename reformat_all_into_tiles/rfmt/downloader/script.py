import sys
import os
import shutil
import random

def estimate_count(src_dir, suffix, prob):
    actual_src = src_dir + suffix
    all_folders = os.listdir(actual_src)
    print(f"Found {len(all_folders)} items in {suffix}. Expecting to copy over approximately {len(all_folders) * prob} items.")

def copy_in_folder(src_dir, dst_dir, suffix, prob):
    actual_src = src_dir + suffix
    actual_dst = dst_dir + suffix

    all_folders = os.listdir(actual_src)
    
    #print(f"Found {len(all_folders)} items in {suffix}. Expecting to copy over approximately {len(all_folders) * prob} items.")
    for folder in all_folders:
        if random.random() < prob:
            src_folder = actual_src + folder
            print(f"\tCopying {src_folder}")
            shutil.copytree(src_folder, actual_dst + folder)


if __name__ == "__main__":
    tile_size = 256
    seq_len = 8

    if(len(sys.argv) < 5):
        print("Usage: <path to folder of files> <path to place to output everything> <train prob> <test prob> <fullres prob>")
        exit(-1)

    src_dir = sys.argv[1] + "/"
    dst_dir = sys.argv[2] + "/"

    estimate_count(src_dir, "rt_train/", float(sys.argv[3]))
    estimate_count(src_dir, "rt_test/", float(sys.argv[4]))
    estimate_count(src_dir, "test_fullres_dir/", float(sys.argv[5]))

    copy_in_folder(src_dir, dst_dir, "rt_train/", float(sys.argv[3]))
    copy_in_folder(src_dir, dst_dir, "rt_test/", float(sys.argv[4]))
    copy_in_folder(src_dir, dst_dir, "test_fullres_dir/", float(sys.argv[5]))

    print("Done!")