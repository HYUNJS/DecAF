import os
import os.path as osp
import zipfile
from multiprocessing import Pool, cpu_count
import tempfile
import shutil
from tqdm import tqdm
import argparse


def zip_files(path, temp_zip_file):
    with zipfile.ZipFile(temp_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(path):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, path)
                zipf.write(full_path, arcname=relative_path)

def main():
    parser = argparse.ArgumentParser(description='Zip files')
    parser.add_argument('dir', help='path to the directory')
    parser.add_argument('out', help='path to the output zip file')
    args = parser.parse_args()
    path      = args.dir[:-1] if args.dir.endswith('/') else args.dir
    out_file  = args.out

    if "mevis" in path:
        dataset = "mevis"
    # elif "refytvos" in path:
    elif "ytvos" in path:
        dataset = "refytvos"

    num_cpus = cpu_count()

    p = Pool(processes=num_cpus)
    subdirs = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    subdir2zip = dict()
    pbar = tqdm(subdirs, desc='Zipping directories to temporary zip files')
    for subdir in subdirs:
        temp_zip_file = tempfile.mkstemp(suffix=f'_{os.getpid()}.zip')[1]
        p.apply_async(zip_files, args=(subdir, temp_zip_file), error_callback=lambda e: print(e), callback=lambda _: pbar.update())
        subdir2zip[subdir] = temp_zip_file
    p.close()
    p.join()
    
    pbar = tqdm(total=len(subdirs), desc='Merging temporary zip files to final zip file')
    with zipfile.ZipFile(out_file, 'w') as final_zipf:
        for subdir, temp_zip_file in subdir2zip.items():
            subdir_name = osp.basename(subdir)
            with zipfile.ZipFile(temp_zip_file, 'r') as temp_zipf:
                for file in temp_zipf.namelist():
                    filepath = osp.join(subdir_name, file)
                    filepath = osp.join('Annotations', filepath) if dataset == "refytvos" else filepath
                    final_zipf.writestr(filepath, temp_zipf.read(file))
            os.remove(temp_zip_file)
            pbar.update()
    pbar.close()

"""
    ## ref-ytvos
    python evaluation/zip_refvos_results.py outputs/refytvos_valid/decaf-v0-sam2-inf-1/ outputs/decaf-v0-sam2-inf-1.zip
    ## mevis
    python evaluation/zip_refvos_results.py outputs/mevis_valid/decaf-v0-sam2-inf-1/ outputs/mevis_valid/decaf-v0-sam2-inf-1.zip
"""
if __name__ == '__main__':
    main()