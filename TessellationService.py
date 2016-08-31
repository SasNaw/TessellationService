#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO delete this:
# /home/sawn/Studium/Master/microservices/TessellationService/TessellationService.py
# -r 200 200 -f -o /home/sawn/Studium/Master/microservices/TessellationService/test /home/sawn/Studium/Master/microservices/TessellationService/img/CMU-1.svs

from openslide import OpenSlide
from PIL import Image
import json
import sys
import numpy as np
import os, os.path
import argparse


PATH = ''
TILE_SIZE = 256
global OUTPUT
OUTPUT = None
global FORCE
FORCE = False
global RESIZE
RESIZE = None


# ======================================   UTILITY   ======================================

def read_json(path):
    try:
        with open(path, 'r') as file:
            str = (file.read())
            data = json.loads(str.decode('utf-8'))
            return data
    except IOError:
        print('Could not load saved annotations from ' + path)


def save_image(image, region, slide_name):
    if(OUTPUT):
        dest = OUTPUT + region['name']
    else:
        dest = region['name']
    if not os.path.exists(dest):
        os.makedirs(dest)
    name = dest + '/' + slide_name + '_' + str(region['uid'])
    if not FORCE:
        while os.path.isfile(name):
            name = name + "_copy"
    if not RESIZE is None:
        print("resize me! :D")
    image.save(name, 'jpeg')
    with open(name + '.context', 'w+') as file:
        data = json.dumps(region.get('context'), ensure_ascii=False)
        file.write(data.encode('utf-8'))


# ======================================   DZI   ======================================


# ======================================   WSI   ======================================

def is_suppoted(file):
    ext = (file.split('.'))[-1]
    if(
        'bif' in ext or
        'mrxs' in ext or
        'npdi' in ext or
        'scn' in ext or
        'svs' in ext or
        'svslide' in ext or
        'tif' in ext or
        'tiff' in ext or
        'vms' in ext or
        'vmu' in ext
    ):
        return 1
    else:
        return 0


def get_bounding_box(region):
    x_min = sys.float_info.max
    x_max = sys.float_info.min
    y_min = x_min
    y_max = x_max
    for coordinate in region.get('imgCoords'):
        x = coordinate.get('x')
        y = coordinate.get('y')
        if(x >= x_max):
            x_max = x
        elif(x < x_min) :
            x_min = x
        if(y >= y_max):
            y_max = y
        elif(y < y_min) :
            y_min = y

    return {'x_max': int(np.ceil(x_max)), 'x_min': int(np.floor(x_min)),
            'y_max': int(np.ceil(y_max)), 'y_min': int(np.floor(y_min))}


def stitch_image(bounding_box, slide):

    # img_width = int(np.ceil((bounding_box['x_max'] - bounding_box['x_min']) / TILE_SIZE) * TILE_SIZE)
    # img_height = int(np.ceil((bounding_box['y_max'] - bounding_box['y_min']) / TILE_SIZE) * TILE_SIZE)
    # img_stitched = Image.new('RGB', (img_width, img_height), color='#ff0000')
    #
    # offset_x = int(np.ceil(bounding_box['x_min'] / TILE_SIZE) * TILE_SIZE)
    # offset_y = int(np.ceil(bounding_box['y_min'] / TILE_SIZE) * TILE_SIZE)
    #
    # for x in range(0, (img_width / 256)):
    #     for y in range(0, (img_height/ 256)):
    #
    #         tile = slide.read_region((x * TILE_SIZE + offset_x, y * TILE_SIZE + offset_y), 0, (TILE_SIZE, TILE_SIZE))
    #         img_stitched.paste(tile, (x*TILE_SIZE, y*TILE_SIZE))

    return img_stitched


def dzi(file):
    regions = read_json(file + '.json')
    for region in regions:
        print("dzi")
    return None


def wsi(file):
    slide = OpenSlide(file)
    slide_name = file.split('/')[-1]
    regions = read_json(file + '.json')
    for region in regions:
        bounding_box = get_bounding_box(region)
        location = (bounding_box['x_min'], bounding_box['y_min'])
        size = (bounding_box['x_max'] - bounding_box['x_min'], bounding_box['y_max'] - bounding_box['y_min'])
        image = slide.read_region(location, 0, size)
        save_image(image, region, slide_name)
    slide.close()


# ======================================   RUN & ARGS   ======================================

def files_from_dir(dir):
    if not dir.endswith('/'):
        dir = dir + '/'
    contents = os.listdir(dir)
    for content in contents:
        if os.path.isdir(dir + content):
            if not content.endswith("_files"):
                files_from_dir(dir + content)
        else:
            regions_from_file(dir + content)


def regions_from_file(file):
    if file.endswith(".dzi"):
        dzi(file)
    else:
        if(is_suppoted(file)):
            wsi(file)


def run(input):
    # input is folder:
    if(os.path.isdir(input)):
        files_from_dir(input)
    # input is file:
    elif(os.path.isfile(input)):
        regions_from_file(input)
    # TODO: list
    else:
        print("ERROR: " + input + " not found!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Either a single file, a list of files or a folder")
    parser.add_argument("-o", "--output", help="directory to store extracted images")
    parser.add_argument("-r", "--resize", help="resize all extracted images to provided width, height in pixel", type=int, nargs=2)
    parser.add_argument("-t", "--tessellate", help="blub", type=int)
    parser.add_argument("-f", "--force-overwrite", help="overwrite images with the same name [False]", action="store_true")

    args = parser.parse_args()

    if(args.resize and args.tessellate):
        print("Only one of -r, -t can be chosen at the same time")
    else:
        FORCE = args.force_overwrite
        RESIZE = args.resize
        OUTPUT = args.output
        if(args.output):
            if not OUTPUT.endswith('/'):
                OUTPUT = OUTPUT + '/'
            if not os.path.exists(OUTPUT):
                os.makedirs(OUTPUT)

        run(args.input)


