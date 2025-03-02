import os
import sys
import shutil
import time
import multiprocessing

def copy_folder(src, dst, counter, lock):
    shutil.copytree(src, dst, dirs_exist_ok=True)

    with lock:
        counter.value += 1

def monitor_copy_progress(suffix, total, counter):
    while counter.value < total:
        print(f"Processed {counter.value}/{total} items for {suffix}")
        time.sleep(1) 
    print(f"Processing for {suffix} complete!")

def copy_items(src_dir, dst_dir, suffix):
    actual_src = src_dir + suffix
    actual_dst = dst_dir + suffix

    folders = os.listdir(actual_src)

    with multiprocessing.Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        
        progress_process = multiprocessing.Process(target=monitor_copy_progress, args=(suffix, len(folders), counter))
        progress_process.start()

        with multiprocessing.Pool() as pool:
            for folder in folders:
                pool.apply_async(copy_folder, args=(actual_src + folder,  actual_dst + folder, counter, lock))
            pool.close()
            pool.join()

        progress_process.join()


if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Usage: <path to folder of files> <path to place to output everything>")
        exit(-1)

    src_dir = sys.argv[1] + "/"
    dst_dir = sys.argv[2] + "/"

    copy_items(src_dir, dst_dir, "rt_train/")
    copy_items(src_dir, dst_dir, "rt_test/")
    copy_items(src_dir, dst_dir, "test_fullres_dir/")