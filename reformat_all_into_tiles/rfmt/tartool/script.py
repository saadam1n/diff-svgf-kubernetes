import sys
import os
import shutil
import tarfile
import multiprocessing
import hashlib

def validate_sample(sample_folder, seq_len):
    if sample_folder.find("Converted") != -1:
        return False

    for i in range(seq_len):
        if not os.path.exists(sample_folder + f"reference{i}.exr"):
            return False

    return True
        
def create_tarball(src_folder, dst_location):
    local_path = f"/tmp/local-{hashlib.sha256(src_folder.encode('utf-8')).hexdigest()}.tgz"
    with tarfile.open(local_path, "w:gz") as tar:
        files = os.listdir(src_folder)
        for file in files:
            tar.add(src_folder + file, file)

    shutil.copyfile(local_path, dst_location)

    os.remove(local_path)

def pool_func(args):
    i, tot, sample_path, dst = args

    print(f"Processing {i}/{tot} items in {suffix}")

    if validate_sample(sample_path, 8):
        create_tarball(sample_path, dst)

    shutil.rmtree(sample_path)

if __name__ == "__main__":
    tile_size = 256
    seq_len = 8

    if(len(sys.argv) < 2):
        print("Usage: <path to folder of files> <path to place to output everything> ")
        exit(-1)

    src_dir = sys.argv[1] + "/"
    dst_dir = sys.argv[2] + "/"

    suffixes = ["rt_train/", "rt_test/", "test_fullres_dir/"]
    for suffix in suffixes:
        sfx_src_dir = src_dir + suffix
        sfx_dst_dir = dst_dir + suffix

        os.makedirs(sfx_dst_dir, exist_ok=True)

        all_folders = os.listdir(sfx_src_dir)
        all_folders = [folder for folder in all_folders if os.path.isdir(os.path.join(sfx_src_dir, folder))]

        pool_args = [(i, len(all_folders), sfx_src_dir + folder + "/", sfx_dst_dir + folder + ".tgz") for i, folder in enumerate(all_folders)]

        with multiprocessing.Pool() as pool:
            pool.map(pool_func, pool_args)

    print("Done!")