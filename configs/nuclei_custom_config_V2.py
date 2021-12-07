# The new config inherits a base config to highlight the necessary modification
_base_ = 'mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco.py'
data_root = '../../DATA/andysu/Nuclei/dataset/'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
# dataset_type = 'nucleiDataset'
dataset_type = 'COCODataset'
classes = ('Nuclei',)
data = dict(
    train=dict(
        img_prefix=data_root + 'all/',
        classes=classes,
        ann_file='./coco_annotations.json'),
    val=dict(
        img_prefix=data_root + 'all/',
        classes=classes,
        ann_file='./coco_annotations_val.json'),
    test=dict(
        img_prefix=data_root + 'test/',
        classes=classes,
        ann_file='./coco_annotations_test.json'))