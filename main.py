import extract
import intro
from imutils import paths 
import argparse
import os
import time

def parseargument():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f','--folder-images', required = True, type = str, help = 'path to folder that contains images')
    return vars(ap.parse_args())
def list_images(path):
    return list(paths.list_images(path))
def main(args):
    img_paths = sorted(list_images(args['folder_images']))
    i = 1
    print('[INFO] Starting ...')
    ls = os.listdir('output/')
    ls = [item.split('.')[0] for item in ls]
    s_t = time.time()
    for path in img_paths:
        name = path.split(os.path.sep)[-1].split('.')[0]
        if name in ls:
            print('[INFO] Processed {}'.format(name))
            i += 1
            continue
        try:
            start = time.time()
            if name < 'K6171-0012':
                intro.main({'image': path, 'intro': 'True'})
            elif name == 'K6171-0012':
                intro.main({'image': path, 'intro': 'False'})
            elif name < 'K6171-0754':
                extract.main({'image': path})
            end = time.time()
            print('[INFO] processing {} - {} - {:.4f}s'.format(i, path.split(os.path.sep)[-1], end - start))
        except:
            print('[INFO] ERROR! {}'.format(path))
        i += 1
    e_t = time.time()
    print('[INFO] Finished. Total: {} images - Total time: {:.4f}s'.format(i - 1, e_t - s_t))
if __name__ == '__main__':
    main(parseargument())