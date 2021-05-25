from __future__ import print_function

import os
import numpy as np
import argparse
import random

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from data import cfg_mnet
from data import AIZOOHeatmapDataset, MoxaHeatmapDataset
from data import preproc_mask_train, preproc_mask_val, detection_heatmap_collate
from layers.modules import MultiBoxMaskHeatmapLoss
from layers.functions.prior_box import PriorBox
from models import FaceMaskDetector

def process_data(args, cfg):

    # parse data path
    dataset_choice = args.dataset_choice
    dataset_root = args.dataset_root

    # parse specific args
    rgb_mean = (104, 117, 123)  # bgr order
    img_dim = cfg['image_size']
    batch_size = cfg['batch_size']
    num_worker = args.num_workers

    # choose dataset
    if dataset_choice == 'AIZOO':
        train_img_dir = dataset_root + '/images'
        train_annotation_dir = dataset_root + '/annotation'
        train_heatmap_dir = dataset_root + '/heatmap'

        dataset_train = AIZOOHeatmapDataset(train_img_dir, train_annotation_dir, train_heatmap_dir,
                                               preproc_mask_train(img_dim, rgb_mean))
        dataset_valid = AIZOOHeatmapDataset(train_img_dir, train_annotation_dir, train_heatmap_dir,
                                               preproc_mask_val(img_dim, rgb_mean))
    elif dataset_choice == 'Moxa3K':
        train_img_dir = dataset_root + '/images'
        train_annotation_dir = dataset_root + '/annotations/Pascal Voc'
        train_heatmap_dir = dataset_root + '/heatmap'

        train_txt_dir = os.path.join(dataset_root, 'train.txt')
        dataset_train = MoxaHeatmapDataset(dataset_root, train_img_dir, train_annotation_dir, train_heatmap_dir, train_txt_dir,
                                    preproc_mask_train(img_dim, rgb_mean))
        dataset_valid = MoxaHeatmapDataset(dataset_root, train_img_dir, train_annotation_dir, train_heatmap_dir, train_txt_dir,
                                    preproc_mask_val(img_dim, rgb_mean))
    else:
        raise Exception('Dataset Not Implemented Error.')

    # obtain training indices that will be used for validation
    num_data = len(dataset_train)
    indices = list(range(num_data))
    # np.random.shuffle(indices)
    seed = 4
    random.Random(seed).shuffle(indices)
    valid_size = 0.1
    split = int(np.floor(valid_size * num_data))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_worker, collate_fn=detection_heatmap_collate)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_worker, collate_fn=detection_heatmap_collate)

    return train_loader, valid_loader


# Setup model
def setup_model(args, cfg):

    # parse arguments
    num_classes = cfg['num_classes']
    img_dim = cfg['image_size']
    num_gpu = cfg['ngpu']
    gpu_train = cfg['gpu_train']
    initial_lr = args.lr

    # Model
    net = FaceMaskDetector(cfg=cfg)
    print("Printing net...")
    print(net)

    # Pretrain weights
    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = torch.load(args.resume_net)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()

    cudnn.benchmark = True

    # set optimizer and scheduler
    optimizer = optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20)

    # set loss
    criterion = MultiBoxMaskHeatmapLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    return net, optimizer, scheduler, criterion, priors


# Train
def train(args, cfg, train_loader, net, optimizer, criterion, priors):

    # loss
    train_loss_total = 0.0
    train_loss_loc = 0.0
    train_loss_cla = 0.0
    train_loss_heat = 0.0

    # Train
    net.train()
    for data, target, heatmap in train_loader:
        # GPU
        data= data.cuda()
        target = [anno.cuda() for anno in target]
        heatmap = heatmap.cuda()
        # clear gradients
        optimizer.zero_grad()
        # forward
        output_train, featuremap = net(data)
        # loss
        loss_l, loss_c, loss_h = criterion(output_train, priors, target, featuremap, heatmap)
        loss = cfg['loc_weight'] * loss_l + loss_c + cfg['heatmap_weight'] * loss_h
        # backward
        loss.backward()
        # perform optimization
        optimizer.step()
        # update traing loss
        train_loss_total += loss.item()
        train_loss_loc += loss_l.item()
        train_loss_cla += loss_c.item()
        train_loss_heat += loss_h.item()

    train_loss = [train_loss_total, train_loss_loc, train_loss_cla, train_loss_heat]

    return train_loss




# Validate
def valid(args, cfg, valid_loader, net, optimizer, criterion, priors):

    # loss
    valid_loss_total = 0.0
    valid_loss_loc = 0.0
    valid_loss_cla = 0.0
    valid_loss_heat = 0.0

    # Validate
    net.eval()  # Fix model difference between training and inference
    with torch.no_grad():  # No gradient update
        for data, target, heatmap in valid_loader:
            # GPU
            data = data.cuda()
            target = [anno.cuda() for anno in target]
            heatmap = heatmap.cuda()
            # forward
            output_valid, featuremap = net(data)
            # loss
            loss_l, loss_c, loss_h = criterion(output_valid, priors, target, featuremap, heatmap)
            loss = cfg['loc_weight'] * loss_l + loss_c + cfg['heatmap_weight'] * loss_h
            # update traing loss
            valid_loss_total += loss.item()
            valid_loss_loc += loss_l.item()
            valid_loss_cla += loss_c.item()
            valid_loss_heat += loss_h.item()

    # loss
    valid_loss = [valid_loss_total, valid_loss_loc, valid_loss_cla, valid_loss_heat]

    return valid_loss


# Process result
def process_result(args, cfg, net, train_loader, valid_loader, epoch, train_loss, valid_loss, valid_loss_min, writer):

    max_epoch = cfg['epoch']
    save_folder = args.save_folder

    # calculate average loss
    train_loss_total = train_loss[0]
    train_loss_loc = train_loss[1]
    train_loss_cla = train_loss[2]
    train_loss_heat = train_loss[3]
    valid_loss_total = valid_loss[0]
    valid_loss_loc = valid_loss[1]
    valid_loss_cla = valid_loss[2]
    valid_loss_heat = valid_loss[3]

    # calculate average loss
    train_size = cfg['batch_size'] * len(train_loader)
    val_size = cfg['batch_size'] * len(valid_loader)

    # calculate average loss
    train_loss_total /= train_size
    train_loss_loc /= train_size
    train_loss_cla /= train_size
    train_loss_heat /= train_size
    valid_loss_total /= val_size
    valid_loss_loc /= val_size
    valid_loss_cla /= val_size
    valid_loss_heat /= val_size

    # print
    print(
        'Epoch:{}/{} || Train Total:{:.3f} Loc:{:.3f} Cla:{:.3f} Heat:{:.3f} || Valid Total:{:.3f} Loc:{:.3f} Cla:{:.3f} Heat:{:.3f}'
            .format(epoch, max_epoch, train_loss_total, train_loss_loc, train_loss_cla, train_loss_heat, valid_loss_total, valid_loss_loc,
                    valid_loss_cla, valid_loss_heat))

    # indicative name
    save_name_indicator = cfg['name'] +  args.indicative_info  # every string should have a '_' in the front

    # save model
    if valid_loss_total < valid_loss_min:
        print(
            'Validation loss decreased ({:.6f} -> {:.6f}). Saving model...'.format(valid_loss_min, valid_loss_total))
        torch.save(net.state_dict(), save_folder + save_name_indicator + '_Best.pth')
        valid_loss_min = valid_loss_total

    if epoch % 50 == 0:
        torch.save(net.state_dict(), save_folder + save_name_indicator + '_' + str(epoch) + '.pth')

    if epoch == max_epoch - 1:
        torch.save(net.state_dict(), save_folder + save_name_indicator + '_Final.pth')

    # tensorboad
    writer.add_scalars('Train Loss', {'Train loss total': train_loss_total,
                                'Train loss location': train_loss_loc,
                                'Train loss classification': train_loss_cla,
                                'Train loss Heatmap': train_loss_heat
                                      }, epoch)

    # tensorboad
    writer.add_scalars('Valid Loss', {'Validation loss total': valid_loss_total,
                                'Validation loss location': valid_loss_loc,
                                'Validation loss classification': valid_loss_cla,
                                'Validation loss heatmap': valid_loss_heat
                                      }, epoch)

    return valid_loss_min



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train face mask detector')
    parser.add_argument('--dataset_choice', default='AIZOO', help='Dataset name')
    parser.add_argument('--dataset_root', default='../../../Data/Face_Mask_Detection/AIZOO/train',
                        help='Dataset root directory')
    parser.add_argument('--network', default='mobilenet0.25', help='Backbone network mobilenet0.25')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='Resume iter for retraining')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    parser.add_argument('--indicative_info', default='_rcam_heatmap', help='Information for different results')

    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    cfg = None
    if args.network == "mobilenet0.25":
        cfg = cfg_mnet
    else:
        raise Exception('Model Not Implemented Error.')

    max_epoch = cfg['epoch']

    # visualization
    writer = SummaryWriter()  # Save to default folders with dates and time

    # Process data
    train_loader, valid_loader = process_data(args, cfg)

    # Setup model
    net, optimizer, scheduler, criterion, priors = setup_model(args, cfg)

    valid_loss_min = np.Inf  # track change in validation loss

    print("Training and Validating")

    for epoch in range(1, max_epoch):

        # train the network
        print('Epoch:{}/{} '.format(epoch, max_epoch))
        train_loss = train(args, cfg, train_loader, net, optimizer, criterion, priors)

        # Valid the network
        # print('Epoch:{}/{} Validating ...'.format(epoch, max_epoch))
        valid_loss = valid(args, cfg, valid_loader, net, optimizer, criterion, priors)

        # Learning Rate Decay
        scheduler.step(valid_loss[0])  # 0 is the total loss

        # Process result
        valid_loss_min = process_result(args, cfg, net, train_loader, valid_loader, epoch,
                       train_loss, valid_loss, valid_loss_min, writer)




