import sys
import os
import shutil
import multiprocessing

def copy_tarball(args):
    src, dst = args
    if src.find("infinigen") != -1:
        print(f"\tCopying {src}")
        shutil.copyfile(src, dst)


if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Usage: <path to folder of files> <path to place to output everything>")
        exit(-1)

    src_dir = sys.argv[1] + "/"
    dst_dir = sys.argv[2] + "/"

    for suffix in ["rt_train/", "rt_test/", "test_fullres_dir/"]:

        actual_src = src_dir + suffix
        actual_dst = dst_dir + suffix

        if os.path.exists(actual_dst):
            shutil.rmtree(actual_dst)
        os.makedirs(actual_dst)

        all_tarballs = os.listdir(actual_src)
        
        pool_args = [(actual_src + tarball, actual_dst + tarball) for tarball in all_tarballs]
        with multiprocessing.Pool() as pool:
            pool.map(copy_tarball, pool_args)


    print("Done!")