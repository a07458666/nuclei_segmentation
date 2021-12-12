import csv
import json
import cv2
import numpy as np
import pycocotools.mask as maskUtils
import os
import argparse
import imagesize
import base64
from tqdm import tqdm
from skimage import measure


def create_test():
    test_data = {"images": [], "categories": []}
    with open("../../DATA/Nuclei/dataset/test_img_ids.json") as f:
        read_data = json.load(f)
        test_data["images"] = read_data
    test_data["categories"] = []
    category = {}
    category["id"] = 1
    category["name"] = "Nuclei"
    test_data["categories"].append(category)

    with open("coco_annotations_test.json", "w") as outfile:
        json.dump(test_data, outfile)


def maskToannotations(data, mask_path, imgdata):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Rescale masks to be same size as associated images, and binarize masks
    img = cv2.resize(img, (imgdata["width"], imgdata["height"]))
    img = img.clip(max=1)

    gt_binary_mask = img
    fortran_gt_binary_mask = np.asfortranarray(gt_binary_mask)
    encoded_ground_truth = maskUtils.encode(fortran_gt_binary_mask)
    ground_truth_area = maskUtils.area(encoded_ground_truth)
    ground_truth_bounding_box = maskUtils.toBbox(encoded_ground_truth)
    contours = measure.find_contours(gt_binary_mask, 0.5)

    # RLE segmentation
    # m = maskUtils.encode(np.asfortranarray(img))
    # area = int(maskUtils.area(m))
    # x, y, w, h = maskUtils.toBbox(m)
    # m['counts'] = base64.b64encode(m['counts']).decode('utf-8')
    category_id = 1

    annotation = {}
    annotation["segmentation"] = []  # m #contours
    annotation["area"] = ground_truth_area.tolist()
    annotation["iscrowd"] = 0
    annotation["image_id"] = imgdata["id"]
    annotation["bbox"] = ground_truth_bounding_box.tolist()
    annotation["category_id"] = category_id
    annotation["id"] = len(data["annotations"])
    annotation["file_name"] = imgdata["file_name"]
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
    data["annotations"].append(annotation)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert open images segmentation data to coco format."
    )
    parser.add_argument(
        "--data_path",
        "-d",
        default="train_e",
        help="Folder that contains the images (as jpg)",
    )
    parser.add_argument(
        "--images",
        "-i",
        default="train_e",
        help="Folder that contains the images (as jpg)",
    )
    parser.add_argument(
        "--masks",
        "-m",
        default="train-masks-e",
        help="Folder that contains the masks (as png)",
    )
    parser.add_argument(
        "--classes",
        "-c",
        default="class-descriptions-boxable.csv",
        help="CSV file that contains class id mappings",
    )
    parser.add_argument(
        "--annotations",
        "-a",
        default="train-annotations-object-segmentation.csv",
        help="CSV file that contains information about annotations",
    )
    parser.add_argument(
        "--remove_unknown_masks",
        action="store_true",
        help="Masks from classes not specified removed from the mask folder",
    )
    parser.add_argument(
        "--generate_yolact_config",
        action="store_true",
        help="Generates json file with custom dataset class",
    )

    args = parser.parse_args()

    data_path = args.data_path
    image_path = args.images
    mask_path = args.masks
    create_test()
    # Setup basic json structure

    data = {"images": [], "annotations": [], "categories": []}

    category = {}
    category["id"] = 1
    category["name"] = "Nuclei"
    data["categories"].append(category)

    # Read images
    print("Reading images")
    imgdata_map = {}
    directory = os.fsencode(data_path)
    index = 1
    files = os.listdir(directory)
    num_imgs = len(files)
    #     files = files[:22]
    print("Files", num_imgs)
    for file in tqdm(files):
        filename = os.fsdecode(file)
        filepath = os.path.join(data_path, filename)
        img = os.listdir(os.path.join(filepath, "images"))[0]
        masks = os.listdir(os.path.join(filepath, "masks"))
        img_path = os.path.join(filepath, "images", img)
        # print('img', img)
        # print('masks', masks)
        # print(img_path)

        if img.endswith(".png"):
            width, height = imagesize.get(img_path)
            fn = filename
            imgdata_map[fn] = {"id": index, "width": width, "height": height}
            image = {}
            image["file_name"] = filename + ".png"
            image["height"] = height
            image["width"] = width
            image["id"] = index
            # print("image = ", image)
            data["images"].append(image)

            index += 1
            # if index > MAX_IMAGE_COUNT:
            #    break

            if index % 1000 == 0:
                print(f"Img read progress: {(index / num_imgs * 100):.2f} %")
            # Mask
            for mask in masks:
                if mask.endswith(".png"):
                    path = os.path.join(filepath, "masks", mask)
                    maskToannotations(data, path, image)

    print("Input data processed, writing json")

    with open("coco_annotations_all.json", "w") as outfile:
        json.dump(data, outfile)

    print("Done")
