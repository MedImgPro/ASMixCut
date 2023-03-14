import argparse
import os
from datetime import *
from multiprocessing import Pool
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from torch.autograd import Variable
from torch.optim import lr_scheduler
import augment.transforms as transforms
cudnn.benchmark = True
from TwoClsModels.resnet_mixup import resnet18,resnet34,resnet50
from TwoClsModels.senet2 import Senet
from TwoClsModels.inceptionv3 import inceptionv3
from helper import AverageMeter
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from dataset import load_data


# #GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6" # 0, 1, 2, ...

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152','efficientnet_b4'
]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

###data
parser.add_argument('-data', default='/public/zouqingqing/ImageNet-master/Train_h/',help='path to dataset')
parser.add_argument('-preddata', default='/public/zouqingqing/ImageNet-master/2-cls/CLS_N_AS_2_val/',help='path to dataset')
parser.add_argument('-extdata', default='/public/zouqingqing/ImageNet-master/2-cls/EXT_VAL/',help='path to dataset')
parser.add_argument('-dataIndexPath', default='/public/zouqingqing/ImageNet-master/2-cls/ASP-O/five_folds/',help='path to dataset index')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')

# training
parser.add_argument('--ngpu', type=int, default=3, help='0 = CPU')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=0.0001, help='weight decay (L2 penalty)')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[150, 225],
                    help='decrease learning rate at these epochs')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--epochs', default=300000

                    , type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--k', '--k_fold', default=5, type=int, metavar='K',
                    help='k_fold (default: 4)')   # k_fold
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.set_defaults(pretrained=True)
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default=''
                                        , type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--train',
                    type=str,
                    default='mixup',
                    choices=['vanilla', 'mixup', 'mixup_hidden'],
                    help='mixup layer')
parser.add_argument('--in_batch',
                    type=str2bool,
                    default=False,
                    help='whether to use different lambdas in batch')
parser.add_argument('--mixup_alpha', default=1.0, type=float, help='alpha parameter for mixup')
parser.add_argument('--dropout',
                    type=str2bool,
                    default=False,
                    help='whether to use dropout or not in final layer')


# Puzzle Mix
parser.add_argument('--box', type=str2bool, default=True, help='true for CutMix')
parser.add_argument('--graph', type=str2bool, default=False, help='true for PuzzleMix')
parser.add_argument('--neigh_size',
                    type=int,
                    default=2,
                    help='neighbor size for computing distance beteeen image regions')
parser.add_argument('--n_labels', type=int, default=3, help='label space size')

parser.add_argument('--beta', type=float, default=1.2, help='label smoothness')
parser.add_argument('--gamma', type=float, default=0.5, help='data local smoothness')
parser.add_argument('--eta', type=float, default=0.2, help='prior term')

parser.add_argument('--transport', type=str2bool, default=True, help='whether to use transport')
parser.add_argument('--t_eps', type=float, default=0.8, help='transport cost coefficient')
parser.add_argument('--t_size',
                    type=int,
                    default=-1,
                    help='transport resolution. -1 for using the same resolution with graphcut')

parser.add_argument('--adv_eps', type=float, default=10.0, help='adversarial training ball')
parser.add_argument('--adv_p', type=float, default=0.0, help='adversarial training probability')

parser.add_argument('--clean_lam', type=float, default=0.0, help='clean input regularization')
parser.add_argument('--mp', type=int, default=8, help='multi-process for graphcut (CPU)')

filepath='/public/zouqingqing/ImageNet-master/TwoClsMainResult/CLS_ASP_O/ae_general_fusion/f5/'
best_prec1 = 0.0

default_conf = {
    'manual_seed': 0,
    'transformer':{
        'train':{
            'raw': [
                {'name': 'RandomFlip'},
                {'name': 'RandomRotate90'},
                {'name': 'RandomRotate2D', 'angle_spectrum': 20, 'mode': 'reflect'},
            ],
        },
        'test':{
            'raw': None,
        }
    }
}

criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.2,1.0])).float()).cuda()
criterion_batch = nn.CrossEntropyLoss(reduction='none', weight=torch.from_numpy(np.array([1.2, 1.0])).float()).cuda()
bce_loss = nn.BCELoss( ).cuda()
bce_loss_sum = nn.BCELoss(reduction='sum').cuda()
softmax = nn.Softmax(dim=1).cuda()

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    elif args.arch == 'inceptionv3':
        model = inceptionv3(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(pretrained=args.pretrained)
    elif args.arch == 'densenet169':
        model = densenet169(pretrained=args.pretrained)
    elif args.arch == 'densenet201':
        model = densenet201(pretrained=args.pretrained)
    elif args.arch == 'densenet161':
        model = densenet161(pretrained=args.pretrained)
    elif args.arch == 'vgg11':
        model = vgg11(pretrained=args.pretrained)
    elif args.arch == 'vgg13':
        model = vgg13(pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
    elif args.arch == 'vgg19':
        model = vgg19(pretrained=args.pretrained)
    elif args.arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg13_bn':
        model = vgg13_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg19_bn':
        model = vgg19_bn(pretrained=args.pretrained)
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
        # for name, child in model.named_children():
        #
        #     if name in ['fc']:
        #         for param in child.parameters():
        #             param.requires_grad = True
        #     else:
        #         for param in child.parameters():
        #             param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features , 2)  # 想输出为2个类别时
        model.fc_H = nn.Linear(model.fc_H.in_features, 2)  # 想输出为2个类别时
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features * 2, 2)  # 想输出为2个类别时
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features * 3, 2)  # 想输出为2个类别时
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features * 3, 2)  # 想输出为2个类别时
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    elif args.arch == 'senet':
        model = Senet(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features * 3, 2)  # 想输出为2个类别时
    else:
        raise NotImplementedError
    print(model)
    # # use cuda
    model = model.cuda()
    #多GPU
    model = torch.nn.DataParallel(model)
    args.num_classes = 2
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    if args.mp > 0:
        mp = Pool(args.mp)
    else:
        mp = None
    checkpoint_dir = os.path.join(filepath,str(args.k), args.arch)
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs', time_stamp), flush_secs=2)
    loss_k, train_k, valid_k = k_fold(args.k,args.epochs,  args.batch_size, optimizer, criterion, model,scheduler,writer,args,mp)
    print('%d-fold validation: min loss rmse %.5f, max train rmse %.5f,max test rmse %.5f' % (
    args.k, loss_k, train_k, valid_k))
    print("Congratulations!!! hou bin")
    writer.close()

def get_default_conf():
    return default_conf.copy()

def k_fold(k_id,epochs,batch_size,optimizer,criterion,model,scheduler,writer,args,mp):
    Ktrain_min_l = []
    Ktrain_acc_max_l = []
    Ktest_acc_max_l = []

    # ####显示数据查验
    # dataiter = iter(train_loader)
    # images1, images2, images3, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images1))

    conf = get_default_conf()

    train_loader, val_loader = load_data(args,k_id,conf)

    args.mean = torch.tensor([0.5]*3 , dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
    args.std = torch.tensor([0.5] *3, dtype=torch.float32).reshape(1, 3, 1, 1).cuda()

    loss_min, train_acc_max, test_acc_max = train(k_id, train_loader, val_loader,model, criterion, optimizer,
                                                  args.epochs,  scheduler,writer,args,mp)
    Ktrain_min_l.append(loss_min)
    Ktrain_acc_max_l.append(train_acc_max)
    Ktest_acc_max_l.append(test_acc_max)

    return sum(Ktrain_min_l) / len(Ktrain_min_l), sum(Ktrain_acc_max_l) / len(Ktrain_acc_max_l), sum(Ktest_acc_max_l) / len(
    Ktest_acc_max_l)




def train(k_id,  train_loader, val_loader,model, criterion, optimizer, epochs,scheduler,writer,args,mp):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    # start = time.time()
    test_acc_max_l = []
    train_acc_max_l = []
    train_l_min_l = []

    Loss_list = []
    Accuracy_list = []
    test_Loss_list = []
    test_Accuracy_list = []

    Train_AUC=[]
    Test_AUC=[]
    extTestAUC = []
    max_AUC=0.1   #随便设置一个比较小的数

    for epoch in range(epochs):
        train_score_list = []
        train_label_list = []
        #adjust_learning_rate(optimizer, epoch, args.lr)
        train_l_sum, train_acc_sum, test_acc, n = 0.0, 0.0, 0.0, 0
        batch_count = 0

        for j, (input1,input2,input3,target) in enumerate(train_loader):

            optimizer.zero_grad()
            target = target.long().cuda()

            input1 = input1.cuda()
            input2 = input2.cuda()
            input3 = input3.cuda()

            unary1 = None
            unary2 = None
            unary3 = None
            noise1 = None
            noise2 = None
            noise3 = None
            adv_mask1 = 0
            adv_mask2 = 0

            # train with clean images
            if args.train == 'vanilla':
                input_var1, input_var2, input_var3,target_var = Variable(input1), Variable(input2),Variable(input3),Variable(target)
                output, reweighted_target = model(input_var1, input_var2, input_var3, target_var)
                loss = bce_loss(torch.softmax(output,dim=1), reweighted_target)

            # train with mixup images
            elif args.train == 'mixup':
                # process for Puzzle Mix
                if args.graph:
                    # whether to add adversarial noise or not
                    if args.adv_p > 0:
                        adv_mask1 = np.random.binomial(n=1, p=args.adv_p)
                        adv_mask2 = np.random.binomial(n=1, p=args.adv_p)
                    else:
                        adv_mask1 = 0
                        adv_mask2 = 0

                    # random start
                    if (adv_mask1 == 1 or adv_mask2 == 1):
                        noise1 = torch.zeros_like(input1).uniform_(-args.adv_eps / 255.,
                                                                 args.adv_eps / 255.)
                        input_orig1 = input1 * args.std + args.mean
                        input_noise1 = input_orig1 + noise1
                        input_noise1 = torch.clamp(input_noise1, 0, 1)
                        noise1 = input_noise1 - input_orig1
                        input_noise1 = (input_noise1 - args.mean) / args.std
                        input_var1 = Variable(input_noise1, requires_grad=True)


                        noise2 = torch.zeros_like(input2).uniform_(-args.adv_eps / 255.,
                                                                 args.adv_eps / 255.)
                        input_orig2 = input2 * args.std + args.mean
                        input_noise2 = input_orig2 + noise2
                        input_noise2 = torch.clamp(input_noise2, 0, 1)
                        noise2 = input_noise2 - input_orig2
                        input_noise2 = (input_noise2 - args.mean) / args.std
                        input_var2 = Variable(input_noise2, requires_grad=True)

                        noise3 = torch.zeros_like(input3).uniform_(-args.adv_eps / 255.,
                                                                 args.adv_eps / 255.)
                        input_orig3 = input3 * args.std + args.mean
                        input_noise3 = input_orig3 + noise3
                        input_noise3 = torch.clamp(input_noise3, 0, 1)
                        noise3 = input_noise3 - input_orig3
                        input_noise3 = (input_noise3 - args.mean) / args.std
                        input_var3 = Variable(input_noise3, requires_grad=True)
                    else:
                        input_var1 = Variable(input1, requires_grad=True)
                        input_var2 = Variable(input2, requires_grad=True)
                        input_var3 = Variable(input3, requires_grad=True)
                    target_var = Variable(target)

                    # calculate saliency (unary)
                    if args.clean_lam == 0:
                        model.eval()
                        output = model(input_var1,input_var2,input_var3)
                        loss_batch = criterion_batch(output, target_var)
                    else:
                        model.train()
                        output = model(input_var1,input_var2,input_var3)
                        loss_batch = 2 * args.clean_lam * criterion_batch(output,
                                                                          target_var) / args.num_classes

                    loss_batch_mean = torch.mean(loss_batch, dim=0)
                    loss_batch_mean.backward(retain_graph=True)

                    unary1 = torch.sqrt(torch.mean(input_var1.grad ** 2, dim=1))
                    unary2 = torch.sqrt(torch.mean(input_var2.grad ** 2, dim=1))
                    unary3 = torch.sqrt(torch.mean(input_var3.grad ** 2, dim=1))

                    # calculate adversarial noise
                    if (adv_mask1 == 1 or adv_mask2 == 1):
                        noise1 += (args.adv_eps + 2) / 255. * input_var1.grad.sign()
                        noise1 = torch.clamp(noise1, -args.adv_eps / 255., args.adv_eps / 255.)
                        adv_mix_coef1 = np.random.uniform(0, 1)
                        noise1 = adv_mix_coef1 * noise1


                        noise2 += (args.adv_eps + 2) / 255. * input_var2.grad.sign()
                        noise2 = torch.clamp(noise2, -args.adv_eps / 255., args.adv_eps / 255.)
                        adv_mix_coef2 = np.random.uniform(0, 1)
                        noise2 = adv_mix_coef2 * noise2

                        noise3 += (args.adv_eps + 2) / 255. * input_var3.grad.sign()
                        noise3 = torch.clamp(noise3, -args.adv_eps / 255., args.adv_eps / 255.)
                        adv_mix_coef3 = np.random.uniform(0, 1)
                        noise3 = adv_mix_coef3 * noise3
                    if args.clean_lam == 0:
                        model.train()
                        optimizer.zero_grad()

                input_var1, input_var2, input_var3, target_var = Variable(input1), Variable(input2), Variable(
                    input3), Variable(target)

                # perform mixup and calculate loss
                # H = H.detach()
                output, reweighted_target,output1_d, output2_d, output3_d, d1, d2, d3, h_output, loss_dg= model(input_var1,
                                                  input_var2,
                                                  input_var3,
                                                  target_var,
                                                  mixup=True,
                                                  args=args,
                                                  grad1=unary1,
                                                  grad2=unary2,
                                                  grad3=unary3,
                                                  noise1=noise1,
                                                  noise2=noise2,
                                                  noise3=noise3,
                                                  adv_mask1=adv_mask1,
                                                  adv_mask2=adv_mask2,
                                                  mp=mp)

                loss = bce_loss(torch.softmax(output,dim=1), reweighted_target)+ bce_loss(torch.softmax(h_output,dim=1), reweighted_target)



            # for manifold mixup
            elif args.train == 'mixup_hidden':
                input_var1, input_var2, input_var3, target_var = Variable(input1), Variable(input2), Variable(
                    input3), Variable(target)
                output, reweighted_target = model(input_var1, input_var2, input_var3, target_var, mixup_hidden=True, args=args)

                loss = bce_loss(torch.softmax(output,dim=1), reweighted_target)
            else:
                raise AssertionError('wrong train type!!')

            out = torch.softmax(output,dim=1)

            train_l_sum += loss.cpu().item()

            losses.update(loss.item(), input1.size(0))


            loss.backward()


            optimizer.step()

            train_acc_sum += (output.argmax(dim=1) == target).sum().cpu().item()

            n += target.shape[0]
            batch_count += 1


            train_score_list.extend(out.detach().cpu().numpy())  #统计sigmoid的输出  1维向量
            train_label_list.extend(target.cpu().numpy())

            if j % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, j, len(train_loader), loss=losses))


        train_score_array = np.array(train_score_list)

        # # 将label转换成onehot形式
        train_label_tensor = torch.tensor(train_label_list)
        train_label_tensor = train_label_tensor.reshape((train_label_tensor.shape[0], 1))
        train_label_onehot = torch.zeros(train_label_tensor.shape[0], 2)
        train_label_onehot.scatter_(dim=1, index=train_label_tensor, value=1)
        train_label_onehot = np.array(train_label_onehot)


        trainAUC=roc_auc_score(train_label_onehot,train_score_array)
        print('trianAUC:',trainAUC)
        Train_AUC.append(trainAUC)

        train_loss=train_l_sum / batch_count
        lr_1 = optimizer.param_groups[0]['lr']
        print('learn_rate : %.15f' % lr_1)


        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('learning rate', lr_1, epoch)
        writer.add_scalar('AUC',trainAUC, epoch)
        try:
            with open(filepath+'train'+str(k_id)+'_'+'results.txt', 'w') as file:
                file.write('Epoch: [{0}]\t'                  
                           'Loss {loss.avg:.4f}'.format(
                    epoch,  loss=losses))
        except Exception as err:
            print(err)
        test_acc, test_l, testAuc, test_score_array, test_label_onehot = validate(
            k_id, val_loader, model, criterion, args.print_freq)

        scheduler.step(testAuc)
        Y_pred = [np.argmax(y) for y in test_score_array]  # 取出y中元素最大值所对应的索引
        Y_valid = [np.argmax(y) for y in test_label_onehot]
        print('Y_valid:',Y_valid)
        print('Y_pred:',Y_pred)
        writer.add_scalar('val loss', test_l, epoch)
        writer.add_scalar('val acc', test_acc, epoch)
        writer.add_scalar('val auc', testAuc, epoch)

        if  testAuc > max_AUC:
            np.savetxt(filepath + "train_score" + str(k_id) + "_" + "array.txt", train_score_array)
            np.savetxt(filepath + "train_label" + str(k_id) + "_" + "onehot.txt", train_label_onehot)
            print("score_array:", test_score_array.shape)  # (batchsize, classnum)
            print("label_onehot:", test_label_onehot.shape)  # torch.Size([batchsize, classnum])
            np.savetxt(filepath + "score" + str(k_id) + "_" + "array.txt", test_score_array)
            np.savetxt(filepath + "label" + str(k_id) + "_" + "onehot.txt", test_label_onehot)
            max_AUC= testAuc
            checkpoint = {
                'epoch': epochs,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'best_auc': testAuc,
                 'train_auc':trainAUC,
                ' train_acc':train_acc_sum / n,
                'test_acc':test_acc,
                'test_auc': testAuc,

        }
            torch.save(checkpoint, filepath + "mri_models" + "_" + str(k_id)+"_" + str(epoch))


        train_l_min_l.append(train_l_sum / batch_count)
        train_acc_max_l.append(train_acc_sum / n)
        test_acc_max_l.append(test_acc)


        print('fold %d epoch %d, avg_train loss %.4f, avg_test loss %.4f, train acc %.3f, test acc %.3f'
            % (k_id, epoch + 1,  train_loss, test_l, train_acc_sum / n, test_acc))
        Loss_list.append(train_l_sum / batch_count)
        Accuracy_list.append(train_acc_sum / n)
        test_Loss_list.append(test_l)
        test_Accuracy_list.append(test_acc)


    index_max = test_acc_max_l.index(max(test_acc_max_l))
    print('fold %d, train_loss_min %.4f, train acc max%.4f, test acc max %.4f'
          % (
          k_id, train_l_min_l[index_max], train_acc_max_l[index_max], test_acc_max_l[index_max]))

    #存储损失数组形式
    train_accuracy_array=np.array(Accuracy_list)
    np.save(filepath+'train'+str(k_id)+'_'+'accuracy_list.txt',train_accuracy_array)
    test_Accuracy_array=np.array(test_Accuracy_list)
    np.save(filepath+'test'+str(k_id)+'_'+'accuracy_list.txt',test_Accuracy_array)
    train_loss_array=np.array(Loss_list)
    np.save(filepath+'train' + str(k_id) + '_' + 'avg_loss_list.txt',train_loss_array)
    test_Loss_array=np.array(test_Loss_list)
    np.save(filepath+'test'+str(k_id)+'_'+'avg_loss_list.txt',test_Loss_array)


    #存储auc
    np.save(filepath+'test'+str(k_id)+'_auc.txt', Test_AUC)
    np.save(filepath+'train' + str(k_id) + '_auc.txt', Train_AUC)

    return train_l_min_l[index_max], train_acc_max_l[index_max], test_acc_max_l[index_max]



def validate(k_id,val_loader,model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        test_l_sum, acc_sum, n = 0.0, 0.0, 0
        batch_count = 0
        # end = time.time()

        score_list = []  # 存储预测得分
        label_list = []  # 存储真实标签

        correct = 0.0
        total = 0.0

        for i, (input1,input2,input3,target) in enumerate(val_loader):
            target = target.cuda()
            input1 = input1.cuda()
            input2 = input2.cuda()
            input3 = input3.cuda()
            input_var1 = Variable(input1)
            input_var2 = Variable(input2)
            input_var3 = Variable(input3)
            target_var = Variable(target)
            # compute output
            output  = model(input_var1,input_var2,input_var3)
            test_loss = criterion(output, target_var)
            out = torch.softmax(output,dim=1)
            #取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(output.data, 1)
            #print(predicted)
            total += target.size(0)
            correct += (predicted == target).sum()

            test_l_sum += test_loss.cpu().item()
            losses.update(test_loss.item(), input1.size(0))


            score_list.extend(out.detach().cpu().numpy())
            label_list.extend(target.cpu().numpy())

            n += target.shape[0]
            batch_count += 1

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    i, len(val_loader), loss=losses))

    print('softmax 测试分类准确率为：%.3f%%' % (100 * correct / total))
    acc = correct / total
    score_array = np.array(score_list)

    # # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], 2)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    testAuc = roc_auc_score(label_onehot,score_array)
    print('testAUC:',testAuc)
    print('batch_count:',batch_count)


    return acc, test_l_sum /batch_count,testAuc,score_array,label_onehot



if __name__ == '__main__':
    main()
