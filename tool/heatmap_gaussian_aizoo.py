import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

epsilon = 1e-6


def read_image(img_dir, annotation_dir):
    # img_dir - what directory are the images in
    # annotation_dir - what directory are the annotations in

    img_names = os.listdir(img_dir)  # list all files in the img folder
    img_names.sort()  # order the images alphabetically
    file_names = np.copy(img_names)
    img_names = [os.path.join(img_dir, img_name) for img_name in img_names]  # join folder and file names

    annotation_names = os.listdir(annotation_dir)  # list all annotation files
    annotation_names.sort()  # order annotation files alphabetically
    annotation_names = [os.path.join(annotation_dir, ann_name) for ann_name in
                        annotation_names]  # join folder and file names

    return img_names, annotation_names, file_names


def get_annotation(annotation_name):   # (num_obj, 5)
    #
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
            annotation[0, -1] = 1  # face
        else:
            annotation[0, -1] = 2  # face with mask

        annotations = np.append(annotations, annotation, axis=0)

    target = np.array(annotations)

    return target


def box_scale(boxes, target_size, img_size):
    boxes[:, 0] = (target_size[1] * (boxes[:, 0] / img_size[1])).astype(np.int)
    boxes[:, 1] = (target_size[0] * (boxes[:, 1] / img_size[0])).astype(np.int)
    boxes[:, 2] = (target_size[1] * (boxes[:, 2] / img_size[1])).astype(np.int)
    boxes[:, 3] = (target_size[0] * (boxes[:, 3] / img_size[0])).astype(np.int)

    return boxes


def convert_corner2center(boxes, cla):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    num_face = len(cla[cla == 1])
    num_mask = len(cla[cla == 2])
    num_box = boxes.shape[0]

    assert num_face + num_mask == num_box

    res = np.zeros((num_face + 2*num_mask, 4))  # (cx, cy, w, h)

    # face part
    res[:num_box, :2] = ((boxes[:num_box, 2:] + boxes[:num_box, :2]) / 2).astype(np.int)
    res[:num_box, 2:] = boxes[:num_box, 2:] - boxes[:num_box, :2]

    j = 0
    for i in range(cla.shape[0]):
        if cla[i] == 2:
            # mask part
            res[num_box+j, 0] = int((boxes[i, 2] + boxes[i, 0]) / 2)  # cx
            cy = ((boxes[i, 3] + boxes[i, 1]) / 2)
            res[num_box+j, 1] = int((cy + boxes[i, 3]) / 2)  # cy
            res[num_box+j, 2] = boxes[i, 2] - boxes[i, 0]  # w
            res[num_box+j, 3] = int((boxes[i, 3] - boxes[i, 1]) / 2)  # h
            j += 1
    return res


def gaussian_radius(boxes):
    return boxes[:, 2:] / 6  # (w, h)


def gaussian_heatmap(center=(2, 2), image_size=(10, 10), sig=1):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    x_axis = np.linspace(0, image_size[0] - 1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1] - 1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel


def multi_gaussian_multi_object_heatmap(center, image_size, sig):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value  (w, h)
    :param cla: classes, 1-face, 2-mask
    :return:
    """

    num_object = len(center)

    kernel = np.zeros(image_size)
    for i in range(num_object):
        x_axis = np.linspace(0, image_size[0] - 1, image_size[0]) - center[i][1]
        y_axis = np.linspace(0, image_size[1] - 1, image_size[1]) - center[i][0]
        yy, xx = np.meshgrid(y_axis, x_axis)

        kernel += np.exp(-0.5 * (np.square(yy) / (np.square(sig[i][0] + epsilon)) + np.square(xx) / (
                    np.square(sig[i][1]) + epsilon)))
    kernel[kernel > 1] = 1

    return kernel


def main():
    # dir
    dataset_root = '../../Data_Bank/Face_Detection/AIZOO/Split/train'
    image_path = dataset_root + '/image'
    annotation_path = dataset_root + '/annotation'
    save_path = dataset_root + '/heatmap_add_mobilenet_middle/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # read image
    img_names, annotation_names, file_names = read_image(image_path, annotation_path)

    # iterate each image
    #target_size = (105, 105) # resnet50
    target_size = (40, 40) # mobilenet0.25
    num_sample = len(img_names)
    # num_sample = 4
    for index in range(num_sample):
        # get image
        img_name = img_names[index]  # get the path of the image at that index
        img = cv2.imread(img_name)  # open the image using the path
        file_name = file_names[index]

        # get annotation
        annotation_name = annotation_names[index]  # get the path to the label file
        target = get_annotation(annotation_name)  # (num_obj, 5)

        # plot bounding box
        for b in target:
            b = list(map(int, b))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        # plt.imshow(img)
        # plt.show()

        # generate heatmap
        img_size = img.shape[:2]
        target = box_scale(target, target_size, img_size)
        center_loc = convert_corner2center(target[:, :4], target[:, -1])
        print(img_name, center_loc)
        sigma = gaussian_radius(center_loc)
        heatmap = multi_gaussian_multi_object_heatmap(center=center_loc[:, :2], image_size=target_size, sig=sigma)

        output_img = heatmap * 255
	
	# save image
        save_name = save_path + str(file_name)
        cv2.imwrite(save_name, output_img)


if __name__ == '__main__':
    main()


