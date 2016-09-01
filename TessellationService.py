#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO delete this:
# /home/sawn/Studium/Master/microservices/TessellationService/TessellationService.py
# -o /home/sawn/Studium/Master/microservices/TessellationService/test /home/sawn/Studium/Master/microservices/TessellationService/img/CMU-1.svs

from openslide import OpenSlide
from PIL import Image
import json
import sys
import numpy as np
import os, os.path
import argparse
import cv2
import xml.etree.ElementTree as ET

TILE_SIZE = 256
global OUTPUT
OUTPUT = None
global FORCE
FORCE = False
global RESIZE
RESIZE = None
WIDTH = 0
HEIGHT = 1
global TESSELLATE
TESSELLATE = None
global SHOW
SHOW = False


# ======================================   UTILITY   ======================================

def read_json(path):
    try:
        with open(path, 'r') as file:
            str = (file.read())
            data = json.loads(str.decode('utf-8'))
            return data
    except IOError:
        print('Could not load saved annotations from ' + path)


def save_image(image, region, slide_name, *tiles):
    if(OUTPUT):
        dest = OUTPUT + region['name']
    else:
        dest = region['name']
    if not os.path.exists(dest):
        os.makedirs(dest)
    name = dest + '/' + slide_name + '_' + str(region['uid'])
    for entry in tiles:
        name += "_" + str(entry)
    if not FORCE:
        while os.path.isfile(name):
            name = name + "_copy"
    if RESIZE:
        image = image.resize(RESIZE)
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


def resize_bounding_box(bounding_box):
    r_ratio = RESIZE[WIDTH] / float(RESIZE[HEIGHT])
    bb_width = float(bounding_box['x_max'] - bounding_box['x_min'])
    bb_height = float(bounding_box['y_max'] - bounding_box['y_min'])
    bb_ratio = bb_width / bb_height
    if r_ratio == bb_ratio:
        return bounding_box
    else:
        if r_ratio == 1:
            # target is square
            s1 = bb_height/bb_width
            s2 = bb_width/bb_height
            scaled = min(bb_width, bb_height) * max(s1, s2) - min(bb_width, bb_height)
            if(bb_width > bb_height):
                bounding_box['y_min'] -= int(np.floor(scaled/2))
                bounding_box['y_max'] += int(np.ceil(scaled/2))
            else:
                bounding_box['x_min'] -= int(np.floor(scaled/2))
                bounding_box['x_max'] += int(np.ceil(scaled/2))
        elif r_ratio < 1:
            # target is higher than wide
            h_s = 1 / r_ratio
            if bb_height > (bb_width * h_s):
                # adjust width:
                w_new = (bb_height / h_s) - bb_width
                bounding_box['x_min'] -= int(np.floor(w_new/2))
                bounding_box['x_max'] += int(np.ceil(w_new/2))
            else:
                # adjust height:
                h_new = h_s * bb_width - bb_height
                bounding_box['y_min'] -= int(np.floor(h_new/2))
                bounding_box['y_max'] += int(np.ceil(h_new/2))
        else:
            # target is wider than high
            w_s = r_ratio
            if bb_width > (bb_height * w_s):
                # adjust height
                h_new = (bb_width / w_s) - bb_height
                bounding_box['y_min'] -= int(np.floor(h_new/2))
                bounding_box['y_max'] += int(np.ceil(h_new/2))
            else:
                # adjust width:
                w_new = w_s * bb_height - bb_width
                bounding_box['x_min'] -= int(np.floor(w_new/2))
                bounding_box['x_max'] += int(np.ceil(w_new/2))

        # check if bb is big enough
        bb_width = float(bounding_box['x_max'] - bounding_box['x_min'])
        if bb_width < RESIZE[WIDTH]:
            s = RESIZE[WIDTH] / bb_width
            bounding_box = scale_bounding_box(bounding_box, s)
        bb_height = float(bounding_box['y_max'] - bounding_box['y_min'])
        if bb_height < RESIZE[HEIGHT]:
            s = RESIZE[HEIGHT] / bb_height
            bounding_box = scale_bounding_box(bounding_box, s)

        return bounding_box


def scale_bounding_box(bounding_box, scale):
    bb_width = float(bounding_box['x_max'] - bounding_box['x_min'])
    add_w = (bb_width * scale) - bb_width
    bounding_box['x_min'] -= int(np.floor(add_w/2))
    bounding_box['x_max'] += int(np.ceil(add_w/2))

    bb_height = float(bounding_box['y_max'] - bounding_box['y_min'])
    add_h = (bb_height * scale) - bb_height
    bounding_box['y_min'] -= int(np.floor(add_h/2))
    bounding_box['y_max'] += int(np.ceil(add_h/2))

    return bounding_box


def tessellate_wsi(slide, slide_name, region):
    n,m = slide.dimensions
    m = m / TESSELLATE[HEIGHT]
    n = n / TESSELLATE[WIDTH]

    if SHOW:
        ox = 999999
        oy = 999999

    mtrx = np.zeros((m, n))
    for coords in region.get('imgCoords'):
        i = int(coords.get('y') / TESSELLATE[HEIGHT])
        j = int(coords.get('x') / TESSELLATE[WIDTH])
        mtrx[i,j] = 1
        if SHOW:
            if(coords.get('y') < oy):
                oy = coords.get('y')
            if(coords.get('x') < ox):
                ox = coords.get('x')

    lst = []
    for i in xrange(0, m):
        for j in xrange(0, n):
            if(mtrx[i,j]):
                lst.append((j,i))

    contour = np.asarray(lst)
    ref_img = Image.new('RGB', (n,m))
    cv_ref_img = np.array(ref_img)
    cv2.drawContours(cv_ref_img, [contour], 0, (255,255,255), -1)
    if SHOW:
        dbg_img = Image.new('RGB', (n,m))
    for i in xrange(0, m):
        for j in xrange(0, n):
            px = cv_ref_img[i,j]
            if (px == [255, 255, 255]).all():
                location = ((j) * TESSELLATE[WIDTH], (i) * TESSELLATE[HEIGHT])
                size = TESSELLATE
                tile = slide.read_region(location, 0, size)
                save_image(tile, region, slide_name, i, j)
                if SHOW:
                    dbg_img.paste(tile, (j * TESSELLATE[WIDTH] - int(ox), i * TESSELLATE[HEIGHT] - int(oy)))
    if SHOW:
        dbg_img.show()


# def stitch_image(bounding_box, slide):
#
#     # img_width = int(np.ceil((bounding_box['x_max'] - bounding_box['x_min']) / TILE_SIZE) * TILE_SIZE)
#     # img_height = int(np.ceil((bounding_box['y_max'] - bounding_box['y_min']) / TILE_SIZE) * TILE_SIZE)
#     # img_stitched = Image.new('RGB', (img_width, img_height), color='#ff0000')
#     #
#     # offset_x = int(np.ceil(bounding_box['x_min'] / TILE_SIZE) * TILE_SIZE)
#     # offset_y = int(np.ceil(bounding_box['y_min'] / TILE_SIZE) * TILE_SIZE)
#     #
#     # for x in range(0, (img_width / 256)):
#     #     for y in range(0, (img_height/ 256)):
#     #
#     #         tile = slide.read_region((x * TILE_SIZE + offset_x, y * TILE_SIZE + offset_y), 0, (TILE_SIZE, TILE_SIZE))
#     #         img_stitched.paste(tile, (x*TILE_SIZE, y*TILE_SIZE))
#
#     return img_stitched


def get_tile_source(file):
    files_dir = file.replace('.dzi', '_files/')
    layers = os.listdir(files_dir)
    layers.remove('metadata.txt')
    layers = map(int, layers)
    return files_dir + str(max(layers)) + '/'


def get_tiles_from_bounding_box(dzi, bounding_box):
    x_min = bounding_box['x_min'] / dzi['tile_size']
    x_max = bounding_box['x_max'] / dzi['tile_size']
    y_min = bounding_box['y_min'] / dzi['tile_size']
    y_max = bounding_box['y_max'] / dzi['tile_size']

    stitch = Image.new('RGB', ((x_max-x_min+1) * dzi['tile_size'], (y_max-y_min+1) * dzi['tile_size']))

    for i in range(x_min, x_max+1):
        for j in range(y_min, y_max+1):
            tile = Image.open(dzi['tile_source'] + str(i) + '_' + str(j) + '.' + dzi['format'])
            stitch.paste(tile, ((i - x_min) * dzi['tile_size'], (j - y_min) * dzi['tile_size']))
    stitch.save("/home/sawn/Studium/Master/microservices/TessellationService/collected", "jpeg")
    return stitch


def create_image_from_tiles(dzi, bounding_box):
    tile_image = get_tiles_from_bounding_box(dzi, bounding_box)
    offset_x = bounding_box['x_min'] / dzi['tile_size']
    offset_y = bounding_box['y_min'] / dzi['tile_size']

    x_min = bounding_box['x_min'] - (offset_x * dzi['tile_size'])
    x_max = bounding_box['x_max'] - (offset_x * dzi['tile_size'])
    y_min = bounding_box['y_min'] - (offset_y * dzi['tile_size'])
    y_max = bounding_box['y_max'] - (offset_y * dzi['tile_size'])

    return tile_image.crop((x_min, y_min, x_max, y_max))


def dzi(file):
    slide_name = file.split('/')[-1]
    with open(file, 'r') as dzi_file:
        content = dzi_file.read()
    root = ET.fromstring(content)
    dzi = {'tile_size': int(root.get('TileSize')), 'width': int(root[0].get('Width')),
           'height': int(root[0].get('Height')), 'tile_source': get_tile_source(file), 'format': root.get('Format')}
    regions = read_json(file + '.json')

    for region in regions:
        bounding_box = get_bounding_box(region)
        image = create_image_from_tiles(dzi, bounding_box)
        save_image(image, region, slide_name)


def wsi(file):
    slide = OpenSlide(file)
    slide_name = file.split('/')[-1]
    regions = read_json(file + '.json')

    for region in regions:
        if(TESSELLATE):
            tessellate_wsi(slide, slide_name, region)
        else:
            bounding_box = get_bounding_box(region)
            if(RESIZE):
                bounding_box = resize_bounding_box(bounding_box)
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
            if not content.endswith('_files'):
                files_from_dir(dir + content)
        else:
            regions_from_file(dir + content)


def regions_from_file(file):
    if file.endswith('.dzi'):
        dzi(file)
    else:
        if(is_suppoted(file)):
            wsi(file)


def run(input):
    for element in input:
        # input is folder:
        if(os.path.isdir(element)):
            files_from_dir(element)
        # input is file:
        elif(os.path.isfile(element)):
            regions_from_file(element)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="[file] or [directory], can also be a list of both", nargs='*')
    parser.add_argument("-o", "--output", help="directory to store extracted images", metavar='[directory]')
    parser.add_argument("-r", "--resize", help="extracted images have a size of [width] x [height] pixel", type=int, nargs=2, metavar=('[width]', '[height]'))
    parser.add_argument("-t", "--tessellate", help="regions are approximated with tiles with a size of [width] x [height] pixel", type=int, nargs=2, metavar=('[width]', '[height]'))
    parser.add_argument("-f", "--force-overwrite", help="overwrite images with the same name [False]", action="store_true")
    parser.add_argument("-s", "--show-tessellated-image", help="put each tessellated image together and show it (for debugging purposes)", action="store_true")

    args = parser.parse_args()

    FORCE = args.force_overwrite
    RESIZE = args.resize
    OUTPUT = args.output
    TESSELLATE = args.tessellate
    SHOW = args.show_tessellated_image
    if(args.output):
        if not OUTPUT.endswith('/'):
            OUTPUT = OUTPUT + '/'
        if not os.path.exists(OUTPUT):
            os.makedirs(OUTPUT)

    run(args.input)