from PIL import  Image
import os
import cv2 as cv
import numpy as np
import json
from functions import euclidean_dist,  generate_cycle_mask, average_del_min
import scipy.io as scio
import glob
import torch
import torch.nn.functional as F
import sys
sys.path.append('/homeLocal/marcelo/IIM')
mode = 'train'
Root = ''
train_path =  os.path.join(Root,'train_data')


dst_imgs_path = os.path.join(Root,'images')
dst_mask_path = os.path.join(Root,'mask')
dst_json_path = os.path.join(Root,'jsons')

cycle  =False

if  not os.path.exists(dst_mask_path):
    os.makedirs(dst_mask_path)

if  not os.path.exists(dst_imgs_path):
    os.makedirs(dst_imgs_path)

if  not os.path.exists(dst_json_path):
    os.makedirs(dst_json_path)


def resize_images(src_path, shift=0, resize_factor = 2, file_list = None):
    if file_list is None:
        file_list = glob.glob(os.path.join(src_path,'images','*.png'))
    print(len(file_list))
    print('Start resizing images')
    for idx, img_path in enumerate(file_list):
        img_id = img_path.split('/')[-1].split('.')[0]
        # print(img_id)
        img_id =  int(img_id.replace('img', '')) + shift
        img_id = str(img_id).zfill(4)
        dst_img_path = os.path.join(dst_imgs_path, img_id+'.png')
        if os.path.exists(dst_img_path):
            continue
        else:
            img_ori = Image.open(img_path)
            w, h = img_ori.size
            new_w, new_h = w*2, h*2
            p_w, p_h = new_w / 1024, new_h / 768
            if p_w < 1 or p_h < 1:
                if p_w > p_h:
                    new_h = 768
                    new_w = int(new_w / p_h)
                    new_w = (new_w // 16 + 1) * 16
                else:
                    new_h = int(new_h / p_w)
                    new_h = (new_h // 16 + 1) * 16
                    new_w = 1024
            else:
                new_w, new_h = (new_w // 16) * 16, (new_h // 16) * 16
            print(f'Index: {idx}')
            print(img_id)
            print(w, h, new_w, new_h)
            new_img = img_ori.resize((new_w,new_h),Image.BILINEAR)
            new_img.save(dst_img_path, quality=95)



def writer_jsons():

    print('Start writing jsons')
    for idx, img_name in enumerate(os.listdir(dst_imgs_path)):
        ImgInfo = {}
        ImgInfo.update({"img_id":img_name})

        img_id = img_name.split('.')[0]

        dst_json_name = os.path.join(dst_json_path, img_id + '.json')
        print(dst_json_name)
        if os.path.exists(dst_json_name):
            continue
        else:
            imgPath = os.path.join(dst_imgs_path, img_name)
            img = Image.open(imgPath)
            size_map = cv.imread(imgPath.replace('images','size_map'),cv.IMREAD_GRAYSCALE)
            size_map =torch.from_numpy(size_map)
            size_map = F.max_pool2d(size_map[None,None,:,:].float(), (199,199),16,99)
            size_map = F.interpolate(size_map, scale_factor=16).squeeze()
            print(size_map.size())
            size_map = size_map.numpy()

            w, h = img.size
            print(f'Index: {idx}')
            print(img_id)

            img_id = str(int(img_id))

            gt_path = os.path.join(train_path,'ground_truth_localization', 'img' + img_id + '.npy')
            ori_imgPath =  os.path.join(train_path,'images', 'img' + img_id + '.png')

            gtInf = np.load(gt_path, allow_pickle=True)

            # print(gtInf)
            ori_img = Image.open(ori_imgPath)
            ori_w, ori_h = ori_img.size
            print('ori', ori_w, ori_h)
            print('resize',w,h)


            w_rate, h_rate= w/ori_w, h/ori_h
            
            points_tuple = np.where(gtInf==1)
            annPoints = np.asarray([[y_coord, x_coord] for x_coord, y_coord in zip(points_tuple[0], points_tuple[1])])

            if len(annPoints) > 0:
                # print(annPoints)
                annPoints[:, 0]=  annPoints[:, 0] * w_rate
                annPoints[:, 1] = annPoints[:, 1] * h_rate
                annPoints = annPoints.astype(int)

                ImgInfo.update({"human_num": len(annPoints)})
                center_w, center_h = [], []
                xy=[]
                wide,heiht = [],[]
                for head in annPoints:

                    x, y = min(head[0], w-1), min(head[1], h-1)
                    center_w.append(x)
                    center_h.append(y)
                    xy.append([int(head[0]),int(head[1])])

                    if ImgInfo["human_num"] > 4:
                        dists = euclidean_dist(head[None, :], annPoints)
                        dists = dists.squeeze()
                        id = np.argsort(dists)
                        p1_y, p1_x = min(annPoints[id[1]][1], h-  1), min(annPoints[id[1]][0], w - 1)
                        p2_y, p2_x = min(annPoints[id[2]][1], h - 1), min(annPoints[id[2]][0], w - 1)
                        p3_y, p3_x = min(annPoints[id[3]][1], h - 1), min(annPoints[id[3]][0], w - 1)
                        # print(id)
                        # import pdb
                        scale = average_del_min([size_map[y,x], size_map[p1_y, p1_x], size_map[p2_y, p2_x], size_map[p3_y, p3_x]])

                        scale = max(scale,4)
                    else:
                        scale = max(size_map[y, x], 4)
                    # print(x,y, scale)
                    area= np.exp(scale)
                    length  = int(np.sqrt(area))
                    wide.append(length)
                    heiht.append(length)
                ImgInfo.update({"points": xy})

                xywh=[]
                for _,(x, y, x_len, y_len) in enumerate(zip(center_w,center_h,wide,heiht)):
                    # print(x,y,x_len,y_len)

                    x_left_top, y_left_top         = max( int( x - x_len / 2) , 0),   max( int(y - y_len / 2) , 0)
                    x_right_bottom, y_right_bottom = min( int(x +  x_len/ 2), w-1),  min( int(y+  y_len / 2), h-1)
                    xywh.append([x_left_top,y_left_top,x_right_bottom,y_right_bottom])

                ImgInfo.update({"boxes": xywh})
                # print(ImgInfo)

                # plot(center_w, center_h, 'g*')
                # plt.imshow(img)
                # for (x_, y_, w_, h_) in ImgInfo["boxes"]:
                #     plt.gca().add_patch(plt.Rectangle((x_, y_), w_ - x_, h_ - y_, fill=False, edgecolor='r', linewidth=1))
                # plt.show()
            else:
                ImgInfo.update({"human_num": 0})
                ImgInfo.update({"points": []})
                ImgInfo.update({"boxes": []})

            with open(dst_json_name, 'w') as  f:
                json.dump(ImgInfo, f)


def generate_masks():
    file_list = glob.glob(os.path.join(dst_imgs_path,'*.png'))

    print('Start generating masks')
    print(len(file_list))
    for idx, img_path in enumerate(file_list):
        if '.png' in img_path :

            img_id = img_path.split('/')[-1].split('.')[0]
            img_ori = Image.open(img_path)
            w, h = img_ori.size

            print(f'Index: {idx}')
            print(img_id)
            print(w, h)
            mask_map = np.zeros((h, w), dtype='uint8')
            gt_name = os.path.join(dst_json_path, img_id.split('.')[0] + '.json')

            with open(gt_name) as f:
                ImgInfo = json.load(f)

            centroid_list = []
            wh_list = []
            for id,(w_start, h_start, w_end, h_end) in enumerate(ImgInfo["boxes"],0):
                centroid_list.append([(w_end + w_start) / 2, (h_end + h_start) / 2])
                wh_list.append([max((w_end - w_start) / 2, 3), max((h_end - h_start) / 2, 3)])
            # print(len(centroid_list))
            centroids = np.array(centroid_list.copy(),dtype='int')
            wh        = np.array(wh_list.copy(),dtype='int')
            wh[wh>25] = 25
            human_num = ImgInfo["human_num"]
            for point in centroids:
                point = point[None,:]

                dists = euclidean_dist(point, centroids)
                dists = dists.squeeze()
                id = np.argsort(dists)

                for start, first  in enumerate(id,0):
                    if  start>0 and start<5:
                        src_point = point.squeeze()
                        dst_point = centroids[first]

                        src_w, src_h = wh[id[0]][0], wh[id[0]][1]
                        dst_w, dst_h = wh[first][0], wh[first][1]

                        count = 0
                        threshold_w, threshold_h = max(-int(max(src_w,dst_w)/2.),-60), max(-int(max(src_h,dst_h)/2.),-60)
                        # threshold_w, threshold_h = -5,-5
                        while  (src_w+ dst_w)-np.abs(src_point[0]-dst_point[0])>threshold_w and  (src_h+ dst_h)-np.abs(src_point[1]-dst_point[1])>threshold_h:

                            if (dst_w * dst_h) > (src_w * src_h):
                                wh[first][0] = max(int(wh[first][0] * 0.9), 1)
                                wh[first][1] = max(int(wh[first][1] * 0.9), 1)
                                dst_w, dst_h = wh[first][0], wh[first][1]
                            else:
                                wh[id[0]][0] = max(int(wh[id[0]][0]*0.9), 1)
                                wh[id[0]][1] = max(int(wh[id[0]][1]*0.9), 1)
                                src_w, src_h = wh[id[0]][0], wh[id[0]][1]


                            if human_num >=3:
                                dst_point_ = centroids[id[start+1]]
                                dst_w_, dst_h_ = wh[id[start+1]][0], wh[id[start+1]][1]
                                if (dst_w_*dst_h_) > (src_w*src_h) and (dst_w_*dst_h_) > (dst_w*dst_h):
                                    if (src_w+ dst_w_)-np.abs(src_point[0]-dst_point_[0])>threshold_w and  (src_h+ dst_h_)-np.abs(src_point[1]-dst_point_[1])>threshold_h:

                                        wh[id[start+1]][0] = max(int(wh[id[start+1]][0] * 0.9), 1)
                                        wh[id[start+1]][1] = max(int(wh[id[start+1]][1] * 0.9), 1)


                            count+=1
                            if count>50:
                                break
            for (center_w, center_h), (width, height)  in zip (centroids, wh):
                assert (width > 0 and height > 0)

                if (0 < center_w < w) and (0 < center_h < h):
                    h_start = (center_h - height)
                    h_end   = (center_h + height)

                    w_start = center_w - width
                    w_end   = center_w + width
                    #
                    if h_start <0:
                        h_start = 0

                    if h_end >h:
                        h_end = h

                    if w_start<0:
                        w_start =0

                    if w_end >w:
                        w_end = w

                    if cycle:
                        mask = generate_cycle_mask(height,width)
                        mask_map[h_start:h_end, w_start: w_end] = mask

                    else:
                        mask_map[h_start:h_end, w_start: w_end] = 1

            mask_map = mask_map*255
            cv.imwrite(os.path.join(dst_mask_path, img_id+'.png'), mask_map, [cv.IMWRITE_PNG_BILEVEL, 1])

def loc_gt_make(  mode = 'test'):
    txt_path = os.path.join(train_path,f'img_list_{mode}.txt')
    with open(txt_path) as f:
        lines = f.readlines()
    img_ids = []
    for line in lines:
        img_ids.append(line.split('\n')[0])


    count = 0
    print('Start making loc_gt txt file')
    for idx, img_id in enumerate(img_ids):
        print(f'Index: {idx}')
        print(img_id)
        json_path = os.path.join(dst_json_path, img_id.replace('img', '').replace('.png', '').zfill(4) + '.json')
        Box_Info = []
        Box_Info.append(img_id)
        if idx != -1:

            with open(json_path) as f:
                infor = json.load(f)

            Box_Info.append(str(infor['human_num']))
            for id, head in enumerate(infor['boxes']):
                x1, y1, x2, y2 = int(head[0]), int(head[1]), int(head[2]), int(head[3])
                center_x, center_y, w, h = int((x1+x2)/2), int((y1+y2)/2),  int((x2-x1)),int((y2-y1)),
                area = w * h
                if area == 0:
                    count += 1
                    continue

                level_area = 0
                if area >= 1 and area < 10:
                    level_area = 0
                elif area > 10 and area < 100:
                    level_area = 1
                elif area > 100 and area < 1000:
                    level_area = 2
                elif area > 1000 and area < 10000:
                    level_area = 3
                elif area > 10000 and area < 100000:
                    level_area = 4
                elif area > 100000:
                    level_area = 5

                r_small = int(min(w, h) / 2)
                r_large = int(np.sqrt (w * w + h * h) / 2)

                Box_Info.append(str(center_x))
                Box_Info.append(str(center_y))
                Box_Info.append(str(r_small))
                Box_Info.append(str(r_large))
                Box_Info.append(str(level_area))

            # print(Box_Info)
            if mode == 'test':
                mode = 'val'
            with open(os.path.join(Root,  mode + '_gt_loc.txt'), 'a') as f:
                for ind, num in enumerate(Box_Info, 1):
                    if ind < len(Box_Info):
                        f.write(num + ' ')
                    else:
                        f.write(num)
                f.write('\n')

    print(count)
if __name__ == '__main__':

    img_root = os.path.join(train_path, 'images')
    train_list_txt_path = os.path.join(train_path, 'img_list_train.txt')
    valid_list_txt_path = os.path.join(train_path, 'img_list_test.txt')

    file_list = []
    for filename in np.loadtxt(valid_list_txt_path, dtype=str):
        if filename.split('.')[1] == 'png':
            file_list.append(os.path.join(img_root, filename))
    for filename in np.loadtxt(train_list_txt_path, dtype=str):
        if filename.split('.')[1] == 'png':
            file_list.append(os.path.join(img_root, filename))
    file_list.sort()

    #================1. resize images ===================
    resize_images(train_path, 0, file_list = file_list)

    # ================2. size_map ==================
    # from scale_map import main
    # main ('CityPark')

    # ================3. box_level annotations ==================
    # writer_jsons()

    # ================4. masks ==================
    # generate_masks()

    # # ==============5. generate val_loc_gt.txt and train_loc_gt.txt==================
    # loc_gt_make(mode='train')
    # loc_gt_make(mode='test')

    print("task is finished")