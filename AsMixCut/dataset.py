"""
author: Zou Qingqing
date: 2021_10_28 09:36
"""

import numpy as np
import augment.transforms as transforms
import matplotlib.pylab as plt
import torch
import cv2
import pydicom



def load_dicom_image(path):
    ds = pydicom.read_file(path)
    data=ds.pixel_array
    [x, y] = data.shape

    a = int(x / 2 - 112)  # x start
    b = int(x / 2 + 112)  # x end
    c = int(y / 2 - 112)  # y start
    d = int(y / 2 + 112)  # y end
    data = data[a:b, c:d]  # 裁剪图像
    if data.shape != [224, 224]:
        data = cv2.resize(data, (224, 224))
    if np.min(data) == np.max(data):
        data = np.zeros((224, 224))
        return data
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / (np.max(data) - np.min(data))
    return data

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, args, is_train,is_val,is_pred,is_ext_test,root,transformer_config=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root, 'r', encoding="utf-8")  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        self.imgs = imgs
        self.is_train = is_train
        self.is_pred = is_pred
        self.is_val = is_val
        self.is_ext_test = is_ext_test
        self.transformer_config = transformer_config

        if self.is_train  and self.transformer_config is not None:
            self.transformer = transforms.get_transformer(transformer_config, mean=0, std=1, phase='train')  # mean鍜宻td娌＄敤鍒?
            self.raw_transform = self.transformer.raw_transform()



    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容

        feature, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息

        img_size=224

        if self.is_train:
            feature1=load_dicom_image(args.data + feature + '/' + 'T1_1.dcm', img_size)
            feature2 =load_dicom_image(args.data + feature + '/' + 'T1_2.dcm', img_size)
            feature3 =load_dicom_image(args.data + feature + '/' + 'T1_3.dcm', img_size)

            feature4= load_dicom_image(args.data + feature + '/' + 'T2_1.dcm', img_size)
            feature5 = load_dicom_image(args.data + feature + '/' + 'T2_2.dcm', img_size)
            feature6 =load_dicom_image(args.data + feature + '/' + 'T2_3.dcm', img_size)

            feature7= load_dicom_image(args.data + feature + '/' + 'YZ_1.dcm', img_size)
            feature8 = load_dicom_image(args.data + feature + '/' + 'YZ_2.dcm', img_size)
            feature9 =load_dicom_image(args.data + feature + '/' + 'YZ_3.dcm', img_size)

            feature_t1 = np.array([feature1, feature2, feature3])
            feature_t2 = np.array([feature4, feature5, feature6])
            feature_yz = np.array([feature7, feature8, feature9])

            feature_t1 = self.raw_transform(feature_t1)

            feature_t2 = self.raw_transform(feature_t2)

            feature_yz = self.raw_transform(feature_yz)

        elif self.is_pred:
            feature1 = load_dicom_image(args.preddata + feature + '/' + 'T1_1.dcm', img_size)

            feature2 = load_dicom_image(args.preddata + feature + '/' + 'T1_2.dcm', img_size)

            feature3 = load_dicom_image(args.preddata + feature + '/' + 'T1_3.dcm', img_size)

            feature4 = load_dicom_image(args.preddata + feature + '/' + 'T2_1.dcm', img_size)

            feature5 = load_dicom_image(args.preddata + feature + '/' + 'T2_2.dcm', img_size)

            feature6 = load_dicom_image(args.preddata + feature + '/' + 'T2_3.dcm', img_size)

            feature7 = load_dicom_image(args.preddata + feature + '/' + 'YZ_1.dcm', img_size)

            feature8 = load_dicom_image(args.preddata + feature + '/' + 'YZ_2.dcm', img_size)

            feature9 = load_dicom_image(args.preddata + feature + '/' + 'YZ_3.dcm', img_size)


            feature_t1 = np.array([feature1, feature2, feature3])

            feature_t2 = np.array([feature4, feature5, feature6])
            feature_yz = np.array([feature7, feature8, feature9])

        elif self.is_val:
            feature1 = load_dicom_image(args.data + feature + '/' + 'T1_1.dcm', img_size)
            feature2 = load_dicom_image(args.data + feature + '/' + 'T1_2.dcm', img_size)
            feature3 = load_dicom_image(args.data + feature + '/' + 'T1_3.dcm', img_size)


            feature4 = load_dicom_image(args.data + feature + '/' + 'T2_1.dcm', img_size)
            feature5 = load_dicom_image(args.data + feature + '/' + 'T2_2.dcm', img_size)
            feature6 = load_dicom_image(args.data + feature + '/' + 'T2_3.dcm', img_size)


            feature7 = load_dicom_image(args.data + feature + '/' + 'YZ_1.dcm', img_size)
            feature8 = load_dicom_image(args.data + feature + '/' + 'YZ_2.dcm', img_size)
            feature9 = load_dicom_image(args.data + feature + '/' + 'YZ_3.dcm', img_size)

            feature_t1 = np.array([feature1, feature2, feature3])
            feature_t2 = np.array([feature4, feature5, feature6])
            feature_yz = np.array([feature7, feature8, feature9])

        elif self.is_ext_test:
            feature1 = load_dicom_image(args.extdata + feature + '/' + 'T1_1.dcm', img_size)
            feature2 = load_dicom_image(args.extdata + feature + '/' + 'T1_2.dcm', img_size)
            feature3 = load_dicom_image(args.extdata + feature + '/' + 'T1_3.dcm', img_size)

            feature4 = load_dicom_image(args.extdata + feature + '/' + 'T2_1.dcm', img_size)
            feature5 = load_dicom_image(args.extdata + feature + '/' + 'T2_2.dcm', img_size)
            feature6 = load_dicom_image(args.extdata + feature + '/' + 'T2_3.dcm', img_size)

            feature7 = load_dicom_image(args.extdata + feature + '/' + 'YZ_1.dcm', img_size)
            feature8 = load_dicom_image(args.extdata + feature + '/' + 'YZ_2.dcm', img_size)
            feature9 = load_dicom_image(args.extdata + feature + '/' + 'YZ_3.dcm', img_size)

            feature_t1 = np.array([feature1, feature2, feature3])
            feature_t2 = np.array([feature4, feature5, feature6])
            feature_yz = np.array([feature7, feature8, feature9])

        feature_t1 = torch.from_numpy(feature_t1.astype(np.float32))
        feature_t2 =torch.from_numpy(feature_t2.astype(np.float32))
        feature_yz = torch.from_numpy(feature_yz.astype(np.float32))


        return feature_t1, feature_t2,feature_yz, label


    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def load_data(args,k_id,conf):
    train_data = MyDataset(args=args, is_train=True, is_val=False, is_pred=False, is_ext_test=False,
                           root=args.dataIndexPath + 'train' + '_' + str(k_id) + 'k.txt',
                           transformer_config=conf['transformer'])
    val_data = MyDataset(args=args, is_train=False, is_val=True, is_pred=False, is_ext_test=False,
                         root=args.dataIndexPath + 'test' + '_' + str(k_id) + '_' + 'k.txt',
                         transformer_config=conf['transformer'])


    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=5)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=64, num_workers=5)


    return train_loader, val_loader