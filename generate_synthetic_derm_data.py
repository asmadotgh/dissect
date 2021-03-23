import numpy as np
import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm
from utils import *
np.random.seed(0)
from PIL import Image, ImageDraw
import math
from enum import Enum

from prep_data import save_processed_label_file

LEN_FP_SKIN_TONE = 6
LEN_RND_SKIN_TONE = 3
LEN_RND_SIZE = 3
LEN_RND_X_OFFSET = 5
LEN_RND_Y_OFFSET = 5


class LESION(Enum):
    BENIGN = 0
    A = 1
    B = 2
    C = 3
    D = 4


def _same_rbg_color(x, y):
    return (x[0] == y[0]) and (x[1] == y[1]) and (x[2] == y[2])


def _draw_polygon(side, rotation, offset_x, offset_y, size):
    xy = [
        (offset_x + (math.cos(th + rotation) + 1) * size,
         offset_y + (math.sin(th + rotation) + 1) * size)
        for th in [i * (2 * math.pi) / side for i in range(side)]
    ]
    return xy


def generate_img(skin_tone, lesion_type, surgical_marking,
                 rnd_skin_tone,  # 3
                 rnd_size,  # N/A if lesion_type == LESION.D | # 3 ow
                 rnd_x_offset,
                 # 3 if lesion_type == LESION.D (values 1,2,3 produce lesions within the frame for LESION.D) | # 5 ow
                 rnd_y_offset,
                 # 3 if lesion_type == LESION.D (values 1,2,3 produce lesions within the frame for LESION.D) | # 5 ow
                 image_name, image_dir, size=(64, 64)):
    fp_colors = [(247, 207, 177), (232, 181, 143), (210, 159, 124), (188, 121, 82), (165, 94, 43), (60, 30, 27)]

    lesion_colors = [(200, 152, 152), (220, 132, 117), (152, 104, 47), (126, 84, 38), (109, 64, 61),
                     (156, 132, 125)]  # (145, 92, 64)
    skin_color = fp_colors[skin_tone]

    _skin_tone_multipliers = [0.8, 1., 1.25]
    lesion_color = tuple(int(c * _skin_tone_multipliers[rnd_skin_tone]) for c in lesion_colors[skin_tone])
    lesion_size = 20 + (rnd_size - 1) * 3
    lesion_x_offset = 0 + (rnd_x_offset - 2) * 5
    lesion_y_offset = 0 + (rnd_y_offset - 2) * 5
    _darker_coef = 2 / 3
    lesion_color_darker = tuple(int(c * _darker_coef) for c in lesion_color)

    img = Image.new('RGB', size, color=skin_color)
    draw = ImageDraw.Draw(img)
    _start_x_y = 20
    _start_x = _start_x_y + lesion_x_offset
    _start_y = _start_x_y + lesion_y_offset
    _end_x = _start_x + lesion_size
    _end_y = _start_y + lesion_size
    _mid_x = int((_start_x + _end_x) / 2)
    _mid_y = int((_start_y + _end_y) / 2)

    _large_lesion_size = 40
    _large_start_x_y = 10
    _large_start_x = _large_start_x_y + lesion_x_offset
    _large_start_y = _large_start_x_y + lesion_y_offset
    _large_end_x = _large_start_x + _large_lesion_size
    _large_end_y = _large_start_y + _large_lesion_size

    _marker_size = 5

    if lesion_type == LESION.BENIGN.value:
        draw.ellipse((_start_x, _start_y, _end_x, _end_y), fill=lesion_color)
    elif lesion_type == LESION.A.value:
        # currently one one static shape
        draw.pieslice([(_start_x, _start_y), (_end_x, _end_y)], 50, 250, fill=lesion_color)
        draw.pieslice([(_mid_x, _mid_y), (_end_x, _end_y)], 0, 300, fill=lesion_color)
    elif lesion_type == LESION.B.value:
        draw.polygon(_draw_polygon(3, 0, _start_x, _start_y, int(lesion_size / 2)), fill=lesion_color)
        draw.polygon(_draw_polygon(3, 10, _start_x, _start_y, int(lesion_size / 2)), fill=lesion_color)
        draw.polygon(_draw_polygon(3, 320, _mid_x, _mid_y, size=int(lesion_size / 4)), fill=lesion_color)
        draw.polygon(_draw_polygon(3, 0, _start_x, _mid_y, size=int(lesion_size / 4)), fill=lesion_color)
        draw.polygon(_draw_polygon(3, 0, _mid_x, _start_y, size=int(lesion_size / 4)), fill=lesion_color)

    elif lesion_type == LESION.C.value:
        # currently only overlay of two colors + salt and pepper noise
        draw.ellipse((_start_x, _start_y, _end_x, _end_y), fill=lesion_color)
        draw.ellipse((_start_x, _start_y, 35, 35), fill=lesion_color_darker)  # TODO
        # add salt and pepper noise
        tmp_img = np.array(img)
        for i in range(size[0]):
            for j in range(size[1]):
                pixel = tmp_img[i, j]
                if not _same_rbg_color(pixel, skin_color):
                    noise = np.random.choice(3, 1, p=[0.8, 0.1, 0.1])
                    if noise == 1:  # salt
                        tmp_img[i, j] = (255, 255, 255)
                    elif noise == 2:  # pepper
                        tmp_img[i, j] = (0, 0, 0)
        img = Image.fromarray(tmp_img, 'RGB')
        draw = ImageDraw.Draw(img)
    elif lesion_type == LESION.D.value:
        # currently only generating large round lesions
        draw.ellipse((_large_start_x, _large_start_y, _large_end_x, _large_end_y), fill=lesion_color)
    if surgical_marking:
        # currently only using a fixed set location of markings
        if lesion_type == LESION.D.value:
            marker_start_x = _large_start_x - _marker_size
            marker_start_y = _large_start_y - _marker_size
            marker_end_x = _large_end_x
            marker_end_y = _large_end_y
        else:
            marker_start_x = _start_x - _marker_size
            marker_start_y = _start_y - _marker_size
            marker_end_x = _end_x
            marker_end_y = _end_y
        marking_positions = [(marker_start_x, marker_start_y), (marker_start_x, marker_end_y),
                             (marker_end_x, marker_start_y), (marker_end_x, marker_end_y)]
        for pos in marking_positions:
            draw.text(pos, 'X', fill='black')

    img.save(os.path.join(image_dir, image_name), "JPEG")


def _num2fname(inp):
    return str(inp).zfill(7)+'.jpg'


def generate_data(output_dir, attribute='malignant'):  # size
    # factors:
    # skin tone: 1, 2, 3, 4, 5, 6 inspired by fitzpatrick rating
    # A: asymmetrical
    # B: jagged border
    # C: different colors
    # D: large diameter
    # has surgical markings: 0, 1

    img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    counter = 0
    # generating benign cases
    for skin_tone in range(LEN_FP_SKIN_TONE):
        for rnd_skin_tone in range(LEN_RND_SKIN_TONE):
            for rnd_size in range(LEN_RND_SIZE):
                for rnd_x_offset in range(LEN_RND_X_OFFSET):
                    for rnd_y_offset in range(LEN_RND_Y_OFFSET):
                        counter += 1
                        generate_img(skin_tone, 0, 0, rnd_skin_tone,  rnd_size,  rnd_x_offset, rnd_y_offset,
                                     _num2fname(counter), img_dir)
    _benign_counter = counter

    # generating malignant cases
    for lesion_type in [1, 2, 3, 4]:
        for surgical_marking in range(2):
            for skin_tone in range(LEN_FP_SKIN_TONE):
                NUM_SAMPLES_NEEDED = int(LEN_RND_SKIN_TONE * LEN_RND_SIZE * LEN_RND_X_OFFSET * LEN_RND_Y_OFFSET / (4*2))
                if lesion_type == 4:
                    for rnd_skin_tone in range(LEN_RND_SKIN_TONE):
                        for rnd_x_offset in [1, 2, 3]:
                            for rnd_y_offset in [1, 2, 3]:
                                counter += 1
                                generate_img(skin_tone, lesion_type, surgical_marking,
                                             rnd_skin_tone,  1, rnd_x_offset, rnd_y_offset,
                                             _num2fname(counter), img_dir)
                else:
                    rnd_skin_tones = np.random.choice(LEN_RND_SKIN_TONE, NUM_SAMPLES_NEEDED)
                    rnd_sizes = np.random.choice(LEN_RND_SIZE, NUM_SAMPLES_NEEDED)
                    rnd_x_offsets = np.random.choice(LEN_RND_X_OFFSET, NUM_SAMPLES_NEEDED)
                    rnd_y_offsets = np.random.choice(LEN_RND_Y_OFFSET, NUM_SAMPLES_NEEDED)
                    for i in range(NUM_SAMPLES_NEEDED):
                        counter += 1
                        generate_img(skin_tone, lesion_type, surgical_marking,
                                     rnd_skin_tones[i],  rnd_sizes[i],  rnd_x_offsets[i], rnd_y_offsets[i],
                                     _num2fname(counter), img_dir)

    # Saving labels
    df = pd.DataFrame(np.transpose(np.array([[_num2fname(i + 1) for i in range(counter)],
                                             [-1] * _benign_counter + [1] * (counter - _benign_counter)])),
                      columns=['Image_Path', attribute])
    save_processed_label_file(df, output_dir, attribute)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', '-d', type=str, default='synthderm')
    args = parser.parse_args()
    generate_data(os.path.join('data', args.dirname))
