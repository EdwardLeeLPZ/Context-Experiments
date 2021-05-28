import json
import argparse
import os
from PIL import Image
import numpy

def main(args):
    input_dir = args.input_dir
    analyse_masks = args.analyse_masks
    json_dir = os.path.join(input_dir, 'json')
    mask_dir = os.path.join(input_dir, 'mask')

    for json_file_name in os.listdir(json_dir):

        full_json_file_path = os.path.join(json_dir, json_file_name)
        with open(full_json_file_path) as json_file:
            data = json.load(json_file)

        proposals = data['proposals']
        refined_boxes = data['refined_boxes']

        if analyse_masks:
            basename = json_file_name.split('.')[0]
            single_frame_mask_dir = os.path.join(mask_dir, basename)
            masks = []
            for mask_name in os.listdir(single_frame_mask_dir):
                mask_path = os.path.join(single_frame_mask_dir, mask_name)

                mask_image = Image.open(mask_path)
                # convert image to numpy array
                mask = numpy.asarray(mask_image)
                masks.append(mask)

            assert len(masks) == len(refined_boxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--analyse_masks", type=bool, default=False)
    args = parser.parse_args()
    main(args)