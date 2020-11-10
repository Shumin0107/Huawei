import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import datetime
import numpy as np
from tqdm import tqdm
from glob import glob
import torch
from torchvision import transforms

from model.ptsemseg.models.hardnet import hardnet
from utils import seed_filling, extract_bottom_boundary, create_full_color_mask, a2d2_colors, color_codes


def infer_image(net, image_path, out_path, crop_to=None, resize_to=None, resize_back=False):
    """

    :param model_path:
    :param num_classes:
    :param image_path:
    :param out_path:
    :param crop_to: [left, right, top, bottom]
    :param resize_to: [width, bottom]
    :param resize_back: True or False
    :return:
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    org_img = cv2.imread(image_path)
    if crop_to:
        left, right, top, bottom = crop_to
        org_img = org_img[top: bottom, left: right, :]
    org_height, org_width = org_img.shape[:2]
    if resize_to:
        post_img = cv2.resize(org_img, resize_to, interpolation=cv2.INTER_LINEAR)
    else:
        post_img = org_img
    # org_img = org_img[18:, 5:1221, :]
    # cv2.imwrite(os.path.join(out_path, 'img.png'), org_img)
    print('input image shape: {}'.format(post_img.shape))
    img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    out = net(img).squeeze()
    mask = out.argmax(dim=0).squeeze().cpu()
    mask = np.uint8(mask)
    if resize_back:
        mask = cv2.resize(mask, (org_width, org_height), interpolation=cv2.INTER_NEAREST)
    print('mask shape: {}'.format(mask.shape))
    cv2.imwrite(os.path.join(out_path, 'mask.png'), mask)
    road_mask = np.zeros_like(mask)
    road_mask[mask == 0] = 255
    print('road mask shape: {}'.format(road_mask.shape))
    cv2.imwrite(os.path.join(out_path, 'road_mask.png'), road_mask)
    color_mask = create_full_color_mask(mask)
    color_mask = cv2.cvtColor(np.uint8(color_mask), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_path, 'color_mask.png'), color_mask)
    print('color mask shape: {}'.format(color_mask.shape))
    if resize_back:
        color_masked_image = cv2.addWeighted(org_img, 0.6, np.uint8(color_mask), 0.4, 0)
    else:
        color_masked_image = cv2.addWeighted(post_img, 0.6, np.uint8(color_mask), 0.4, 0)
    print('color masked image shape: {}'.format(color_masked_image.shape))
    cv2.imwrite(os.path.join(out_path, 'color_masked_image.png'), color_masked_image)


def infer_image_v5(net, image_dir, mask_dir, crop_to=False, resize_to=False, resize_back=False, crop_back=False,
                   road_only=False, color_masked=False):
    """
    Generate binary road masks.
    Note that the images in the image_dir should have the same size.
    :param net:
    :param image_dir:
    :param mask_dir:
    :param crop_to: (left, right, top, bottom), the width would be right-left, the height would be bottom-top
    :param resize_to: (width, height)
    :param resize_back: if true, resize the output back to the original image size
    :param crop_back: if true, padding the cropped area with a certain value(default value is 0)
    :return:
    """
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    image_list = os.listdir(image_dir)
    image_list.sort()
    for i in tqdm(image_list):
        if '.png' not in i and '.jpg' not in i:
            continue
        img_pth = os.path.join(image_dir, i)
        org_img = cv2.imread(img_pth)
        org_height, org_width = org_img.shape[:2]
        post_img = org_img
        if crop_to != False:
            left, right, top, bottom = crop_to
            post_img = org_img[top:bottom, left:right, :]
        if resize_to != False:
            pre_resized_height, pre_resized_width = post_img.shape[:2]
            post_img = cv2.resize(post_img, resize_to)
        img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        out = net(img).squeeze()

        mask = out.argmax(dim=0).squeeze().cpu()
        mask = np.array(mask, dtype=np.uint8)

        padding = net.n_classes

        ###############################
        # Only for generate road mask #
        ###############################
        if road_only:
            assert not color_masked, 'There should be no color mask with only road!'
            mask[mask == 0] = 255
            mask[mask < 255] = 0
            padding = 0

        # h_tmp, w_tmp = mask.shape
        # mask[:h_tmp//2, :] = 0


        ##################
        # connected area #
        ##################
        # try:
        #     mask_copy = mask.copy()
        #     mask_copy = seed_filling(mask_copy, seed_coord=(365, 621), seed_value=0, connected_value=2)
        #     mask[mask_copy != 2] = 1 # value 1 indicates non road pixels
        #     mask[mask_copy == 2] = 0 # value 0 indicates road pixels
        # except IndexError:
        #     print('{} met an IndexError! Please check it out!'.format(img_pth))

        if resize_back:
            assert resize_to, 'Parameter <resize_to> should have value if <resize_back> is True!'
            mask = cv2.resize(mask, (pre_resized_width, pre_resized_height), interpolation=cv2.INTER_NEAREST)
        if crop_back:
            assert crop_to, 'Parameter <crop_to> should have value if <crop_back> is True!'
            org_sized_mask = np.ones([org_height, org_width], dtype=np.uint8) * padding
            org_sized_mask[top: bottom, left: right] = mask
            mask = org_sized_mask
        if road_only:
            save_name = os.path.join(mask_dir, i[:-4] + '_road.png')
            cv2.imwrite(save_name, mask)
        elif color_masked:
            color_mask = create_full_color_mask(mask, color_codes)
            color_mask = cv2.cvtColor(np.uint8(color_mask), cv2.COLOR_RGB2BGR)
            color_masked_image = cv2.addWeighted(org_img, 0.5, np.uint8(color_mask), 0.5, 0)
            save_name = os.path.join(mask_dir, i[:-4] + '_color_masked.png')
            cv2.imwrite(save_name, color_masked_image)
        else:
            save_name = os.path.join(mask_dir, i[:-4] + '_mask.png')
            cv2.imwrite(save_name, mask)


def images_to_video(image_path_list, image_width, image_height, fps, out_path):
    vwriter = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (image_width, image_height))
    for img_pth in tqdm(image_path_list):
        img = cv2.imread(img_pth)
        vwriter.write(img)
    vwriter.release()


def infer_video_to_video(net, video_path, out_path='./out/out_videos',
                resize_to=None, crop_to=None, resize_back=False, start_frame=0, end_frame=None):
    """
    inference video frames
    :param video_path:
    :param model_path:
    :param num_classes:
    :param resize_to: (width, height)
    :param crop_to: (left, right, top, bottom), the width would be right-left, the height would be bottom-top
    :param resize_back: if true, resize the output back to the original image size
    :param start_frame: start inference from which frame
    :param end_frame: end inference at which frame
    :return: N/A
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    vcapture = cv2.VideoCapture(video_path)
    vcapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    num_frames = vcapture.get(cv2.CAP_PROP_FRAME_COUNT)
    if not end_frame:
        end_frame = num_frames
    print('number of frames: ', num_frames)
    file_name = 'freespace_{:%Y%m%dT%H%M%S}.avi'.format(datetime.datetime.now())
    file_name = os.path.join(out_path, file_name)

    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if crop_to:
        width = crop_to[1] - crop_to[0]
        height = crop_to[3] - crop_to[2]

    if resize_to and not resize_back:
        width, height = resize_to

    print('out width and height: ', width, height)
    vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    # vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    count = start_frame
    success = True
    print('Processing video...')
    tq = tqdm(total=int(end_frame - start_frame))
    while success:
        # read next frame
        success, image = vcapture.read()
        # print(image.shape)
        tq.update(1)
        if success:
            # pre-process image
            if crop_to:
                org_image = image[crop_to[2]:crop_to[3], crop_to[0]:crop_to[1], :]
            else:
                org_image = image

            if resize_to:
                image = cv2.resize(org_image, resize_to)

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            img = img.unsqueeze(0)
            img = img.cuda()
            out = net(img).squeeze()
            out = out.argmax(dim=0).squeeze().int()

            mask = out.cpu().numpy()

            ################################################################
            ## find the connected road component in front of the car head ##
            ################################################################
            # mask = seed_filling(mask, seed_coord=(260, 440), seed_value=0, connected_value=2)
            # mask[mask != 2] = 1 # value 1 indicates non road pixels
            # mask[mask == 2] = 0 # value 0 indicates road pixels

            ######################
            ## extract boundary ##
            ######################
            # boundary = extract_bottom_boundary(mask, pad_num=3, pad_value=1)
            # mask[boundary] = 8 # value 2 indicates road boundary

            if resize_back:
                mask = cv2.resize(np.uint8(mask), dsize=(width, height), interpolation=cv2.INTER_NEAREST)

            ############################
            ## Add fully colored mask ##
            ############################
            color_mask = create_full_color_mask(mask)
            # mask_h, mask_w = mask.shape
            # color_mask = cv2.resize(np.uint8(color_mask), dsize=(mask_w*4, mask_h*4), interpolation=cv2.INTER_LINEAR)
            color_mask = cv2.cvtColor(np.uint8(color_mask), cv2.COLOR_RGB2BGR)

            ###########################
            ## Add colored road mask ##
            ###########################
            # color_mask = create_road_color_mask(mask, road_id=0)

            ####################################
            ## Add colored drivable area mask ##
            ####################################
            # color_mask = create_drivable_color_mask(mask)

            ###############################
            ## Add colored ego lane mask ##
            ###############################
            # color_mask = create_ego_lane_color_mask(mask, ego_lane_id=1)

            # cv2.imwrite(os.path.join(out_pth, 'mask', '{}.jpg'.format(count)), color_mask)
            if resize_back:
                color_masked_image = cv2.addWeighted(org_image, 0.7, np.uint8(color_mask), 0.3, 0)
            else:
                color_masked_image = cv2.addWeighted(image, 0.7, np.uint8(color_mask), 0.3, 0)

            #######################################
            ## post processing for drivable maps ##
            #######################################
            # ego_lane_mask = np.equal(mask, 1).astype(int)
            # bottom_left_point, bottom_right_point, bottom_point, \
            # mid_left_point, mid_right_point, mid_point, \
            # remote_point = ego_lane_mask_filter(ego_lane_mask, bottom=410)
            # color_masked_image = cv2.line(color_masked_image, bottom_left_point, bottom_right_point, (0, 255, 255), 3)
            # color_masked_image = cv2.line(color_masked_image, mid_left_point, mid_right_point, (0, 255, 255), 2)
            # color_masked_image = cv2.line(color_masked_image, bottom_point, mid_point, (0, 0, 255), 2)
            # color_masked_image = cv2.line(color_masked_image, mid_point, remote_point, (0, 0, 255), 2)

            # if resize_to:
            #     color_masked_image = cv2.resize(color_masked_image, (width, height))

            ##########################################
            ## post processing for semantic results ##
            ##########################################
            # cv2.rectangle(color_masked_image, (10, 80), (950, 270), (0, 97, 255), 2)
            # boundary_info = generate_multi_boundary(mask, road_class=3, bottom=60, top=20)
            # for j in range(len(boundary_info) - 1):
            #     pt1 = boundary_info[j]
            #     pt2 = boundary_info[j + 1]
            #     color = boundary_colors[pt1['type']]
            #     cv2.line(color_masked_image, (pt1['width'] * 4, pt1['height'] * 4), (pt2['width'] * 4, pt2['height'] * 4), color, 3)

            vwriter.write(color_masked_image)
            count += 1
            if count == end_frame:
                break
    tq.close()
    vwriter.release()
    print('Video processing is done! Saved to ', file_name)


def infer_video_to_video_v2(net, num_classes, video_path, out_dir, crop_to=(0, 0, 0, 0), resize_to=(0, 0),
                            resize_back=False, crop_back=False, start_frame=0, end_frame=0):
    """
    Generate binary road masks.
    Note that the images in the image_dir should have the same size.
    :param net:
    :param image_dir:
    :param mask_dir:
    :param crop_to: (left, right, top, bottom), the width would be right-left, the height would be bottom-top
    :param resize_to: (width, height)
    :param resize_back: if true, resize the output back to the original image size
    :param crop_back: if true, padding the cropped area with a certain value(default value is 0)
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vcapture = cv2.VideoCapture(video_path)
    vcapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    # fps = 25
    print('frames per second: ', fps)
    num_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('number of frames: ', num_frames)
    org_width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('original video width: ', org_width)
    print('original video height: ', org_height)

    video_width = org_width
    video_height = org_height
    if crop_to != (0, 0, 0, 0) and not crop_back:
        video_width = crop_to[1] - crop_to[0]
        video_height = crop_to[3] - crop_to[2]
    if resize_to != (0, 0) and not resize_back:
        video_width, video_height = resize_to

    file_name = 'freespace_{:%Y%m%dT%H%M%S}.avi'.format(datetime.datetime.now())
    file_name = os.path.join(out_dir, file_name)
    vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (video_width, video_height))

    if end_frame == 0:
        end_frame = num_frames

    count = start_frame
    success = True
    print('Processing video...')
    tq = tqdm(total=int(end_frame - start_frame))
    while success:
        success, org_img = vcapture.read()
        tq.update(1)
        post_img = org_img
        if crop_to != (0, 0, 0, 0):
            pre_cropped_img = post_img
            left, right, top, bottom = crop_to
            post_img = org_img[top:bottom, left:right, :]
        if resize_to != (0, 0):
            pre_resized_img = post_img
            pre_resized_height, pre_resized_width = post_img.shape[:2]
            post_img = cv2.resize(post_img, resize_to)
        img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('./out/img/{}.jpg'.format(count), img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        out = net(img).squeeze()
        print(out.shape)
        mask = out.argmax(dim=0).squeeze().cpu()
        print(mask.shape)
        mask = np.array(mask, dtype=np.uint8)

        ###############################
        # Only for generate road mask #
        ###############################
        # road_mask = mask.copy()
        # road_mask[road_mask == 0] = 255
        # road_mask[road_mask < 255] = 0

        ##################
        # connected area #
        ##################
        # try:
        #     mask_copy = mask.copy()
        #     mask_copy = seed_filling(mask_copy, seed_coord=(365, 621), seed_value=0, connected_value=2)
        #     mask[mask_copy != 2] = 1 # value 1 indicates non road pixels
        #     mask[mask_copy == 2] = 0 # value 0 indicates road pixels
        # except IndexError:
        #     print('{} met an IndexError! Please check it out!'.format(img_pth))

        video_img = post_img
        if resize_back:
            assert resize_to != (0, 0), 'Parameter <resize_to> should have value if <resize_back> is True!'
            mask = cv2.resize(mask, (pre_resized_width, pre_resized_height), interpolation=cv2.INTER_NEAREST)
            video_img = pre_resized_img
        if crop_back:
            assert crop_to != (0, 0, 0, 0), 'Parameter <crop_to> should have value if <crop_back> is True!'
            org_sized_mask = np.ones([org_height, org_width], dtype=np.uint8) * num_classes
            org_sized_mask[top: bottom, left: right] = mask
            mask = org_sized_mask
            video_img = pre_cropped_img
        color_mask = create_full_color_mask(mask, color_codes)
        color_mask = cv2.cvtColor(np.uint8(color_mask), cv2.COLOR_RGB2BGR)

        color_masked_image = cv2.addWeighted(video_img, 0.6, np.uint8(color_mask), 0.4, 0)
        vwriter.write(color_masked_image)

        count += 1
        if end_frame != 0 and count == end_frame:
            break

    tq.close()
    vwriter.release()
    print('Video processing is done! Saved as ', file_name)


def infer_video_to_images(net, num_classes, video_path, out_dir, crop_to=(0, 0, 0, 0), resize_to=(0, 0),
                          resize_back=False, crop_back=False, start_frame=0, end_frame=None):
    """
    Generate binary road masks.
    Note that the images in the image_dir should have the same size.
    :param net:
    :param image_dir:
    :param mask_dir:
    :param crop_to: (left, right, top, bottom), the width would be right-left, the height would be bottom-top
    :param resize_to: (width, height)
    :param resize_back: if true, resize the output back to the original image size
    :param crop_back: if true, padding the cropped area with a certain value(default value is 0)
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vcapture = cv2.VideoCapture(video_path)
    vcapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    # fps = 25
    print('frames per second: ', fps)
    num_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('number of frames: ', num_frames)
    org_width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('original video width: ', org_width)
    print('original video height: ', org_height)

    video_width = org_width
    video_height = org_height
    if crop_to != (0, 0, 0, 0) and not crop_back:
        video_width = crop_to[1] - crop_to[0]
        video_height = crop_to[3] - crop_to[2]
    if resize_to != (0, 0) and not resize_back:
        video_width, video_height = resize_to

    if end_frame is None:
        end_frame = num_frames

    count = start_frame
    success = True
    print('Processing video...')
    tq = tqdm(total=int(end_frame - start_frame))
    while success:
        success, org_img = vcapture.read()
        tq.update(1)
        if count % 100 == 0:
            post_img = org_img
            if crop_to != (0, 0, 0, 0):
                pre_cropped_img = post_img
                left, right, top, bottom = crop_to
                post_img = org_img[top:bottom, left:right, :]
            if resize_to != (0, 0):
                pre_resized_img = post_img
                pre_resized_height, pre_resized_width = post_img.shape[:2]
                post_img = cv2.resize(post_img, resize_to)
            img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            img = img.unsqueeze(0)
            img = img.cuda()
            out = net(img).squeeze()

            mask = out.argmax(dim=0).squeeze().cpu()
            mask = np.array(mask, dtype=np.uint8)
            print(mask)

            ###############################
            # Only for generate road mask #
            ###############################
            # road_mask = mask.copy()
            # road_mask[road_mask == 0] = 255
            # road_mask[road_mask < 255] = 0
            # mask[mask == 0] = 255
            # mask[mask < 255] = 0


            ##################
            # connected area #
            ##################
            # try:
            #     mask_copy = mask.copy()
            #     mask_copy = seed_filling(mask_copy, seed_coord=(365, 621), seed_value=0, connected_value=2)
            #     mask[mask_copy != 2] = 1 # value 1 indicates non road pixels
            #     mask[mask_copy == 2] = 0 # value 0 indicates road pixels
            # except IndexError:
            #     print('{} met an IndexError! Please check it out!'.format(img_pth))

            video_img = post_img
            if resize_back:
                assert resize_to != (0, 0), 'Parameter <resize_to> should have value if <resize_back> is True!'
                mask = cv2.resize(mask, (pre_resized_width, pre_resized_height), interpolation=cv2.INTER_NEAREST)
                # video_img = pre_resized_img
            if crop_back:
                assert crop_to != (0, 0, 0, 0), 'Parameter <crop_to> should have value if <crop_back> is True!'
                org_sized_mask = np.ones([org_height, org_width], dtype=np.uint8) * num_classes
                # org_sized_mask = np.zeros([org_height, org_width], dtype=np.uint8)
                org_sized_mask[top: bottom, left: right] = mask
                mask = org_sized_mask
                video_img = pre_cropped_img
            color_mask = create_full_color_mask(mask, color_codes)
            color_mask = cv2.cvtColor(np.uint8(color_mask), cv2.COLOR_RGB2BGR)
            #
            color_masked_image = cv2.addWeighted(video_img, 0.7, np.uint8(color_mask), 0.3, 0)
            save_name = os.path.join(out_dir, '{}.png'.format(count))
            # cv2.imwrite(save_name, mask)
            cv2.imwrite(save_name, color_masked_image)

        count += 1
        if end_frame != 0 and count == end_frame:
            break

    tq.close()
    print('Video processing is done! Saved to ', out_dir)


if __name__ == '__main__':
    from model.NASv3.erfnet_retrain import Encoder_reset
    from model.NASv3.get_model import switches_cell
    num_classes = 20
    # model_path = './runs/hardnet_catId/deepdrive/hardnet_deepdrive_catId_best_model.pkl'
    # model_path = './runs/hardnet_catId/deepdrive_ohem/hardnet_deepdrive_catId_best_model.pkl'
    # model_path = './runs/hardnet_catId/cityscapes_ohem/hardnet_cityscapes_catId_best_model.pkl'
    # model_path = './runs/hardnet_catId/cityscapes_focal/hardnet_cityscapes_catId_best_model.pkl'
    # model_path = '/extend/l00471718/FCHarDNet/runs/hardnet_catId/deepdrive_focal_finetune_512x1024_v2/hardnet_deepdrive_catId_best_model.pkl'
    # model_path = '/extend/l00471718/FCHarDNet/runs/hardnet_a2d2/a2d2_focal/hardnet_a2d2_best_model.pkl'
    model_path = './model/NASv3/best.pth.tar'
    # model_path = "/home/wx987516/zmz/FCHarDNet-master/weights/hardnet70_cityscapes_model.pkl"
    # video_path = '../dataset/ME630_408_Recorder_video.avi'
    # video_path = './tmp/Manka_rFpro_0615_new.wmv'
    # out_dir = './out/out_videos'
    net = Encoder_reset(num_classes, switches_cell)
    # net = hardnet(num_classes)
    # model_state = torch.load(model_path)['model_state']
    model_state = torch.load(model_path)['state_dict']
    # model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    net.load_state_dict(model_state)
    net.cuda()
    net.eval()

    ######################
    # infer single image #
    ######################
    # image_path = './data/tmp.png'
    # out_path = './tmp'
    # infer_image(net, image_path, out_path, crop_to=(285, 1309, 5, 517), resize_to=(1024, 512), resize_back=True)

    ###########################
    # infering video to video #
    ###########################
    # video_path = '/home/dataset/SoftSimulation/20200813-1/normal/undistort_normal.avi'
    # out_dir = './out/tmp/'
                                                                          # (2048, 1024)
    # infer_video_to_video_v2(net, num_classes, video_path, out_dir, crop_to=(686, 2734, 88, 1112),resize_to=(1024, 512),
    #                         resize_back=True, start_frame=4700, crop_back=True, end_frame=5200)

    ###########################
    # infering go_data/huawei #
    ###########################
    # data_dir = '../go_mono/go_data/wanchuang/PNG'
    # dir_list = os.listdir(data_dir)
    # dir_list.sort()
    # for dir in dir_list:
    #     print(dir)
    #     image_dir = os.path.join(data_dir, dir)
    #     mask_dir = image_dir
    # #     if not os.path.exists(mask_dir):
    # #         os.makedirs(mask_dir)
    #     infer_image_v5(net, image_dir, mask_dir, resize_to=(1024, 576), resize_back=True)

    # image_path_list = glob('./out/slam/byd_dianbo2_crop_road_color_masked_image/*.png')
    # image_path_list.sort()
    # out_path = './out/slam/byd_dianbo2_crop_road_30fps.avi'
    # images_to_video(image_path_list, 1920, 900, 30, out_path)


    ################################
    # inference a single directory #
    ###############################
    image_dir = './cityscapes/'
    mask_dir = './out/cityscapes_crop_nasv3/'
    #(285, 1309, 5, 517)
    infer_image_v5(net, image_dir, mask_dir, crop_to=(0, 2048, 0, 1024), resize_to=(2048, 1024), resize_back=True,
                   crop_back=True, road_only=False, color_masked=True)

    ############################
    # infering video to images #
    ############################
    # video_path = './data/undistort.avi'
    # out_dir = './tmp/undistort_510_2910_0_2100_color'
    # infer_video_to_images(net, num_classes, video_path, out_dir, crop_to=(510, 2910, 0, 1200), resize_to=(1024, 512),
    #                       resize_back=True, crop_back=True, start_frame=0, end_frame=None)

    # image_dir = '../dataset/a2d2/processed_9classes_512x1024/images/val'
    # mask_dir = './tmp/bdd_infer_a2d2'
    #
    # infer_image_v5(net, image_dir, mask_dir)
