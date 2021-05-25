import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from pprint import PrettyPrinter
import matplotlib.pyplot as plt
import pandas as pd

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
dataset_labels = ('face', 'mask')
label_map = {k: v + 1 for v, k in enumerate(dataset_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping



def get_preds(pred_dir):

    preds = os.listdir(pred_dir)
    preds.sort()  # order annotation files alphabetically
    
    boxes = []
    labels = []
    confs = []
    
    for pred in preds:
        if 'txt' in pred:

            filepath = os.path.join(pred_dir, pred)
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
                img_file = lines[0].rstrip('\n\r')
                lines = lines[2:]

            predictions = np.zeros((0, 6))

            for line in lines:
                prediction = np.zeros((1, 6))
                line = line.rstrip('\r\n').split(' ')
                if line[0] == '':
                    continue

                prediction[0, 0] = float(line[0])
                prediction[0, 1] = float(line[1])
                prediction[0, 2] = float(line[2]) + float(line[0])
                prediction[0, 3] = float(line[3]) + float(line[1])

                prediction[0, 4] = float(line[4])  # confidence

                if line[-2] == 'face':
                    prediction[0, 5] = 1
                else:
                    prediction[0, 5] = 2

                predictions = np.append(predictions, prediction, axis=0)
            
            
            predictions = torch.from_numpy(predictions)

            boxes.append(predictions[:, :4])
            labels.append(predictions[:, 4])
            confs.append(predictions[:, 5])
            
    # print(boxes)
    # print(labels)
    # print(confs)

    return boxes, labels, confs


def get_gts(dataset_choice, annotation_dir, txt_dir):
    """
    obtain ground truth
    
    
    """

    if dataset_choice == 'AIZOO':
        annotation_names = os.listdir(annotation_dir)  # list all annotation files
        annotation_names.sort()  # order annotation files alphabetically
        annotation_names = [os.path.join(annotation_dir, ann_name) for ann_name in
                            annotation_names]  # join folder and file names

    elif dataset_choice == 'Moxa3K':
        df = pd.read_csv(txt_dir, header=None)
        img_full_names = df.values
        names = []
        for i in range(img_full_names.shape[0]):
            names.append(img_full_names[i][0][14:])
        names.sort()

        annotation_names = [os.path.join(annotation_dir, ann_name.replace('jpg', 'xml')) for ann_name in
                                names]  # join folder and file names

    len_ann = len(annotation_names)

    bboxes = []
    labels = []

    for index in range(len_ann):

        annotation_name = annotation_names[index]  # get the path to the label file
        annotation_tree = ET.parse(annotation_name)  # use xml parser to load the file

        annotations = np.zeros((0, 5))
        if len(annotation_tree.findall('object')) == 0:
            return annotations

        for object in annotation_tree.findall('object'):

            annotation = np.zeros((1, 5))
            # bbox
            bndbox_xml = object.find("bndbox")
            annotation[0, 0] = float(bndbox_xml.find('xmin').text)
            annotation[0, 1] = float(bndbox_xml.find('ymin').text)
            annotation[0, 2] = float(bndbox_xml.find('xmax').text)
            annotation[0, 3] = float(bndbox_xml.find('ymax').text)
            # name
            if dataset_choice == 'AIZOO':
                if object.find("name").text == 'face':
                    annotation[0, -1] = 1
                else:
                    annotation[0, -1] = 2
            elif dataset_choice == 'Moxa3K':
                if object.find("name").text == 'nomask':
                    annotation[0, -1] = 1
                else:
                    annotation[0, -1] = 2


            annotations = np.append(annotations, annotation, axis=0)

        annotations = torch.from_numpy(annotations)

        bboxes.append(annotations[:, :4])
        labels.append(annotations[:, 4])

    return bboxes, labels


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    rec = rec.cpu()
    prec = prec.cpu()
    
    if use_07_metric:           # VOC 2007 AP metric
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        
        mpre = prec
        mrec = rec
        
    else:
        # correct AP calculation (VOC 2010 and later)
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))  
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]  # different precision points

        # AP = AP1 + AP2 + AP3 + AP4 + ..
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, mrec, mpre


def calculate_mAP_general(det_boxes, det_labels, det_scores, true_boxes, true_labels, plot_pr=True):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    # make sure all lists of tensors of the same length, i.e. number of images
    assert len(det_boxes) == len(det_labels) == len(det_scores) == \
           len(true_boxes) == len(true_labels)
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects), n_objects: total num of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    # true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)

    # list for pr
    modified_precisions = []
    modified_recalls = []

    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        n_class_objects = len(true_class_images)

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((n_class_objects), dtype=torch.uint8)
        true_class_boxes_detected = true_class_boxes_detected.to(device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img, 4)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If this object has already not been detected, it's a true positive
                if true_class_boxes_detected[original_ind] == 0:
                    true_positives[d] = 1
                    true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                # Otherwise, it's a false positive (since this object is already accounted for)
                else:
                    false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)

        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_class_objects  # (n_class_detections)

        # Find average precision
        average_precisions[c - 1], modified_recall, modified_precision = voc_ap(cumul_recall, cumul_precision)

        # Append recall and precision
        modified_recalls.append(modified_recall)
        modified_precisions.append(modified_precision)


    # pr curve
    if plot_pr:
        plt.plot(modified_recalls[0], modified_precisions[0], label='Face')
        plt.plot(modified_recalls[1], modified_precisions[1], label='Mask')
        plt.legend(frameon=False)
        plt.title("PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


if __name__ == '__main__':

    # Get ground truth data

    # AIZOO evaluation
    dataset_choice = 'AIZOO'
    dataset_root = '../../../Data/Face_Mask_Detection/AIZOO/Split/test/'
    annotation_dir = os.path.join(dataset_root, 'annotation')
    true_boxes, true_labels = get_gts(dataset_choice, annotation_dir, txt_dir=None)

    # Moxa3K evaluation
    # dataset_choice = 'Moxa3K'
    # annotation_dir = '../../../Data/Face_Mask_Detection/Moxa3K' + '/annotations/Pascal Voc'
    # txt_dir = '../../../Data/Face_Mask_Detection/Moxa3K' + '/test.txt'
    # true_boxes, true_labels = get_gts(dataset_choice, annotation_dir, txt_dir)

    # Get predicted data
    pred_dir = './results/txt/AIZOO'
    # pred_dir = './results/txt/Moxa3K/'
    det_boxes, det_scores, det_labels = get_preds(pred_dir)

    # Calculate mAP
    APs, mAP = calculate_mAP_general(det_boxes, det_labels, det_scores, true_boxes, true_labels, plot_pr=False)

    # Print AP for each class
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)