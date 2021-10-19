import valer
import argparse
import os

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--pre_train', type=bool, default=True, help='stage 1 or 2')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='nn.Parallel needs or not')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--gpu_ids', type=str, default="0, 1", help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='interval between model checkpoints')
    parser.add_argument('--params_file_path', type=str, default='palm.pdparams', help='the path of model file')
    parser.add_argument('--optimizer_file_path', type=str, default='palm.pdopt', help='the path of optimizer file')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=10, help='size of the batches')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Adam: learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='Adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='Adam: beta 1')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam: beta 2')
    parser.add_argument('--weight_decay', type=float, default=0, help='Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type=int, default=10,
                        help='lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type=float, default=0.5,
                        help='lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_l1', type=float, default=1, help='the parameter of L1Loss')
    parser.add_argument('--lambda_perceptual', type=float, default=10, help='the parameter of perceptual loss')
    parser.add_argument('--lambda_gan', type=float, default=0.1, help='the parameter of GAN loss')
    parser.add_argument('--lambda_classification', type=float, default=0.01, help='the parameter of GAN loss')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--num_classes', type=int, default=1, help='num of classes')
    parser.add_argument('--in_channels', type=int, default=1, help='input RGB image')
    parser.add_argument('--scribble_channels', type=int, default=3, help='input scribble image')
    parser.add_argument('--out_channels', type=int, default=3, help='output RGB image')
    parser.add_argument('--start_channels', type=int, default=64, help='latent channels')
    parser.add_argument('--pad', type=str, default='reflect', help='the padding type')
    parser.add_argument('--activ_g', type=str, default='lrelu', help='the activation type')
    parser.add_argument('--activ_d', type=str, default='lrelu', help='the activation type')
    parser.add_argument('--norm_g', type=str, default='none', help='normalization type')
    parser.add_argument('--norm_d', type=str, default='bn', help='normalization type')
    parser.add_argument('--init_type', type=str, default='xavier', help='the initialization type')
    parser.add_argument('--init_gain', type=float, default=0.02, help='the initialization gain')
    # Dataset parameters
    parser.add_argument('--train_data_path', type=str, default='./data/PALM-Training400',
                        help='the train data folder')
    parser.add_argument('--val_data_path', type=str, default='./data/PALM-Validation400',
                        help='the val data folder')
    parser.add_argument('--val_label_path', type=str, default='./data/PM_Label_and_Fovea_Location.csv',
                        help='the val label folder')
    parser.add_argument('--imgsize', type=int, default=224, help='size of image')

    opt = parser.parse_args()
    print(opt)

    '''
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''

    valer.evaluation(opt)

