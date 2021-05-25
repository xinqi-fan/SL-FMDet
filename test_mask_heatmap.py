from __future__ import print_function

import os
import argparse
import cv2
import pandas as pd
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from data import cfg_mnet
from models.detector import FaceMaskDetector
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms, py_cpu_nms_cross_class
from utils.box_utils import decode
from utils.timer import Timer
from utils.bounding_box import bbox_add


parser = argparse.ArgumentParser(description='Test face mask detector')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_AIZOO_rcam_transfer_heatmap_middle_Best.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobilenet0.25', help='Backbone network mobilenet0.25')
parser.add_argument('--dataset_choice', default='AIZOO', help='Dataset name')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_txt_folder', default='./results/txt', type=str, help='Dir to save txt results')
parser.add_argument('--save_img_folder', default='./results/img', type=str, help='Dir to save img results')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='../../../Data/Face_Mask_Detection/AIZOO/Split/test/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':

    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobilenet0.25":
        cfg = cfg_mnet
    else:
        raise Exception('Model Not Implemented Error.')

    # net and model
    net = FaceMaskDetector(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    if args.dataset_choice == 'AIZOO':
        testset_folder = args.dataset_folder + '/images'
        test_dataset = os.listdir(testset_folder)

    elif args.dataset_choice == 'Moxa3K':
        img_dir = args.dataset_folder + '/images'
        txt_dir = args.dataset_folder + '/test.txt'
        df = pd.read_csv(txt_dir, header=None)
        img_full_names = df.values

        names = []
        for i in range(img_full_names.shape[0]):
            names.append(img_full_names[i][0][14:])

        testset_folder = img_dir
        test_dataset = names

    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    with torch.no_grad():
        for i, img_name in enumerate(test_dataset):

            image_path = os.path.join(testset_folder, img_name)
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = np.float32(img_raw)

            # testing scale
            target_size = 840
            max_size = 840
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            resize = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)
            if args.origin_size:
                resize = 1

            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            # Inference
            _t['forward_pass'].tic()
            (loc, conf), featuremap = net(img)   # forward pass
            _t['forward_pass'].toc()

            _t['misc'].tic()

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores_face = conf.squeeze(0).data.cpu().numpy()[:, 1]
            scores_mask = conf.squeeze(0).data.cpu().numpy()[:, 2]

            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)

            # ignore low scores
            inds_face = np.where(scores_face > args.confidence_threshold)[0]
            inds_mask = np.where(scores_mask > args.confidence_threshold)[0]

            boxes_face = boxes[inds_face]
            scores_face = scores_face[inds_face]

            boxes_mask = boxes[inds_mask]
            scores_mask = scores_mask[inds_mask]

            # keep top-K before NMS
            order_face = scores_face.argsort()[::-1]
            boxes_face = boxes_face[order_face]
            scores_face = scores_face[order_face]

            order_mask = scores_mask.argsort()[::-1]
            boxes_mask = boxes_mask[order_mask]
            scores_mask = scores_mask[order_mask]

            # do NMS
            dets_face = np.hstack((boxes_face, scores_face[:, np.newaxis])).astype(np.float32, copy=False)
            keep_face = py_cpu_nms(dets_face, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets_face = dets_face[keep_face, :]

            dets_mask = np.hstack((boxes_mask, scores_mask[:, np.newaxis])).astype(np.float32, copy=False)
            keep_mask = py_cpu_nms(dets_mask, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets_mask = dets_mask[keep_mask, :]

            dets_face, dets_mask = py_cpu_nms_cross_class(dets_face, dets_mask, thresh=0.80, margin=0.4)

            # dets = np.concatenate((dets, landms), axis=1)
            _t['misc'].toc()

            # --------------------------------------------------------------------

            save_txt_folder = args.save_txt_folder + '/' + args.dataset_choice + '/'
            if not os.path.exists(save_txt_folder):
                os.makedirs(save_txt_folder)
            save_name = save_txt_folder + img_name[:-4] + ".txt"

            with open(save_name, "w") as fd:
                bboxs_face = dets_face
                bboxs_mask = dets_mask
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(bboxs_face)+len(bboxs_mask)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for face_box in bboxs_face:
                    x = int(face_box[0])
                    y = int(face_box[1])
                    w = int(face_box[2]) - int(face_box[0])
                    h = int(face_box[3]) - int(face_box[1])
                    confidence = str(face_box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " face" + " \n"
                    fd.write(line)
                for mask_box in bboxs_mask:
                    x = int(mask_box[0])
                    y = int(mask_box[1])
                    w = int(mask_box[2]) - int(mask_box[0])
                    h = int(mask_box[3]) - int(mask_box[1])
                    confidence = str(mask_box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " mask" + " \n"
                    fd.write(line)

            print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

            # save image
            if args.save_image:
                for b in dets_face:
                    if b[4] < args.vis_thres:
                        continue
                    # text = str("Face:"+"{:.2f}".format(b[4]))
                    # text = "Face"
                    # bbox_add(img_raw, b[0], b[1], b[2], b[3], text, "red")
                    bbox_add(img_raw, b[0], b[1], b[2], b[3], color="red")

                for b in dets_mask:
                    if b[4] < args.vis_thres:
                        continue
                    # text = str("Mask:" + "{:.2f}".format(b[4]))
                    # text = "Mask"
                    # bbox_add(img_raw, b[0], b[1], b[2], b[3], text, "green")
                    bbox_add(img_raw, b[0], b[1], b[2], b[3], color="green")

                # save image
                save_img_folder = args.save_img_folder + '/' + args.dataset_choice + '/'
                if not os.path.exists(save_img_folder):
                    os.makedirs(save_img_folder)
                name = save_img_folder + img_name
                cv2.imwrite(name, img_raw)

                

