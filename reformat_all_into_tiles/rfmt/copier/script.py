import os
import sys
import shutil

def copy_items(src_dir, dst_dir, suffix):
    actual_src = src_dir + suffix
    actual_dst = dst_dir + suffix

    folders = os.listdir(actual_src)

    print(f"Found {len(folders)} items in {suffix}.")
    for i, folder in enumerate(folders):
        print(f"\tCopying item {i}\t/{len(folders)}. Item name is {folder}")
        shutil.copytree(actual_src + folder, actual_dst + folder, dirs_exist_ok=True)

    print(f"Done processing {suffix}")

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Usage: <path to folder of files> <path to place to output everything>")
        exit(-1)

    src_dir = sys.argv[1] + "/"
    dst_dir = sys.argv[2] + "/"

    copy_items(src_dir, dst_dir, "rt_train/")
    copy_items(src_dir, dst_dir, "rt_test/")
    copy_items(src_dir, dst_dir, "test_fullres_dir/")