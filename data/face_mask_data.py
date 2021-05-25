import os
import os.path
import sys
import numpy as np
import torch
import torch.utils.data as data
import cv2
import xml.etree.ElementTree as ET
import pandas as pd


class FaceMaskDataset(data.Dataset):
    def __init__(self, img_dir, annotation_dir, preproc=None):
        self.preproc = preproc

        self.img_dir = img_dir  # what directory are the images in
        self.annotation_dir = annotation_dir  # what directory are the annotations in
        # self.transform = transform  # what transforms were passed to the initialiser

        self.img_names = os.listdir(img_dir)  # list all files in the img folder
        self.img_names.sort()  # order the images alphabetically
        self.img_names = [os.path.join(img_dir, img_name) for img_name in self.img_names]  # join folder and file names

        self.annotation_names = os.listdir(annotation_dir)  # list all annotation files
        self.annotation_names.sort()  # order annotation files alphabetically
        self.annotation_names = [os.path.join(annotation_dir, ann_name) for ann_name in
                                 self.annotation_names]  # join folder and file names

        # print(self.img_names)
        # print(self.annotation_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        img_name = self.img_names[index]  # get the path of the image at that index
        img = cv2.imread(img_name)  # open the image using the path

        annotation_name = self.annotation_names[index]  # get the path to the label file
        annotation_tree = ET.parse(annotation_name)  # use xml parser to load the file

        annotations = np.zeros((0, 5))
        if len(annotation_tree.findall('object')) == 0:
            return annotations

        for object in annotation_tree.findall('object'):

            annotation = np.zeros((1, 5))
            # bbox
            bndbox_xml = object.find("bndbox")
            annotation[0, 0] = int(bndbox_xml.find('xmin').text)
            annotation[0, 1] = int(bndbox_xml.find('ymin').text)
            annotation[0, 2] = int(bndbox_xml.find('xmax').text)
            annotation[0, 3] = int(bndbox_xml.find('ymax').text)
            # name
            if object.find("name").text == 'face':
                annotation[0, -1] = 1
            else:
                annotation[0, -1] = 2
            # annotation[0, -1] = 1

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        # print(f'image name {img_name}')
        # print(f'imgs {img} | targets {target}')

        return torch.from_numpy(img), target


class AIZOOHeatmapDataset(data.Dataset):
    def __init__(self, img_dir, annotation_dir, heatmap_dir, preproc=None):
        self.preproc = preproc

        self.img_dir = img_dir  # what directory are the images in
        self.annotation_dir = annotation_dir  # what directory are the annotations in
        self.heatmap_dir = heatmap_dir
        # self.transform = transform  # what transforms were passed to the initialiser

        self.img_names = os.listdir(img_dir)  # list all files in the img folder
        self.img_names.sort()  # order the images alphabetically
        self.img_names = [os.path.join(img_dir, img_name) for img_name in self.img_names]  # join folder and file names

        self.annotation_names = os.listdir(annotation_dir)  # list all annotation files
        self.annotation_names.sort()  # order annotation files alphabetically
        self.annotation_names = [os.path.join(annotation_dir, ann_name) for ann_name in
                                 self.annotation_names]  # join folder and file names

        self.heatmap_names = os.listdir(heatmap_dir)  # list all files in the heatmap folder
        self.heatmap_names.sort()  # order the images alphabetically
        self.heatmap_names = [os.path.join(heatmap_dir, heatmap_name) for heatmap_name in self.heatmap_names]  # join folder and file names

        # print(self.img_names)
        # print(self.annotation_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        img_name = self.img_names[index]  # get the path of the image at that index
        img = cv2.imread(img_name)  # open the image using the path

        heatmap = cv2.imread(self.heatmap_names[index], cv2.IMREAD_GRAYSCALE)
        heatmap = heatmap / 255.0

        annotation_name = self.annotation_names[index]  # get the path to the label file
        annotation_tree = ET.parse(annotation_name)  # use xml parser to load the file

        annotations = np.zeros((0, 5))
        if len(annotation_tree.findall('object')) == 0:
            return annotations

        for object in annotation_tree.findall('object'):

            annotation = np.zeros((1, 5))
            # bbox
            bndbox_xml = object.find("bndbox")
            annotation[0, 0] = int(bndbox_xml.find('xmin').text)
            annotation[0, 1] = int(bndbox_xml.find('ymin').text)
            annotation[0, 2] = int(bndbox_xml.find('xmax').text)
            annotation[0, 3] = int(bndbox_xml.find('ymax').text)
            # name
            if object.find("name").text == 'face':
                annotation[0, -1] = 1
            else:
                annotation[0, -1] = 2
            # annotation[0, -1] = 1

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        heatmap = np.expand_dims(heatmap, axis=0)

        return torch.from_numpy(img).float(), target, torch.from_numpy(heatmap).float()


class MoxaHeatmapDataset(data.Dataset):
    def __init__(self, root_dir, img_dir, annotation_dir, heatmap_dir, txt_dir, preproc=None):
        self.preproc = preproc

        self.img_dir = img_dir  # what directory are the images in
        self.annotation_dir = annotation_dir  # what directory are the annotations in
        self.heatmap_dir = heatmap_dir
        self.txt_dir = txt_dir

        df = pd.read_csv(self.txt_dir, header=None)
        img_full_names = df.values
        self.names = []
        for i in range(img_full_names.shape[0]):
            self.names.append(img_full_names[i][0][14:])

        self.img_names = [os.path.join(img_dir, img_name) for img_name in self.names]

        if self.heatmap_dir:
            self.heatmap_names = [os.path.join(heatmap_dir, heatmap_name) for heatmap_name in self.names]

        self.annotation_names = [os.path.join(annotation_dir, ann_name.replace('jpg', 'xml')) for ann_name in
                                 self.names] 

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        if self.heatmap_dir:

            img_name = self.img_names[index]  # get the path of the image at that index
            img = cv2.imread(img_name)  # open the image using the path

            heatmap = cv2.imread(self.heatmap_names[index], cv2.IMREAD_GRAYSCALE)
            heatmap = heatmap / 255.0

            annotation_name = self.annotation_names[index]  # get the path to the label file
            annotation_tree = ET.parse(annotation_name)  # use xml parser to load the file

            annotations = np.zeros((0, 5))
            if len(annotation_tree.findall('object')) == 0:
                return annotations

            for object in annotation_tree.findall('object'):

                annotation = np.zeros((1, 5))
                # bbox
                bndbox_xml = object.find("bndbox")
                annotation[0, 0] = int(bndbox_xml.find('xmin').text)
                annotation[0, 1] = int(bndbox_xml.find('ymin').text)
                annotation[0, 2] = int(bndbox_xml.find('xmax').text)
                annotation[0, 3] = int(bndbox_xml.find('ymax').text)
                # name
                if object.find("name").text == 'nomask':
                    annotation[0, -1] = 1
                else:
                    annotation[0, -1] = 2

                annotations = np.append(annotations, annotation, axis=0)

            target = np.array(annotations)
            if self.preproc is not None:
                img, target = self.preproc(img, target)

            heatmap = np.expand_dims(heatmap, axis=0)

            return torch.from_numpy(img).float(), target, torch.from_numpy(heatmap).float()

        else:
            img_name = self.img_names[index]  # get the path of the image at that index
            img = cv2.imread(img_name)  # open the image using the path

            annotation_name = self.annotation_names[index]  # get the path to the label file
            annotation_tree = ET.parse(annotation_name)  # use xml parser to load the file

            annotations = np.zeros((0, 5))
            if len(annotation_tree.findall('object')) == 0:
                return annotations

            for object in annotation_tree.findall('object'):

                annotation = np.zeros((1, 5))
                # bbox
                bndbox_xml = object.find("bndbox")
                annotation[0, 0] = int(bndbox_xml.find('xmin').text)
                annotation[0, 1] = int(bndbox_xml.find('ymin').text)
                annotation[0, 2] = int(bndbox_xml.find('xmax').text)
                annotation[0, 3] = int(bndbox_xml.find('ymax').text)
                # name
                if object.find("name").text == 'nomask':
                    annotation[0, -1] = 1
                else:
                    annotation[0, -1] = 2

                annotations = np.append(annotations, annotation, axis=0)

            target = np.array(annotations)
            if self.preproc is not None:
                img, target = self.preproc(img, target)


            return torch.from_numpy(img).float(), target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        for tup in sample:
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


def detection_heatmap_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    heatmaps = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup) and tup.shape[0] == 3:
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)
            elif torch.is_tensor(tup) and tup.shape[0] == 1:
                heatmaps.append(tup)

    return (torch.stack(imgs, 0), targets, torch.stack(heatmaps, 0))

