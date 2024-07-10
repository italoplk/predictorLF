import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Random seed for np and tf (-1 to avoid seeding)
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    # Training parametes
    parser.add_argument('--dataset-path', type=str, default='/scratch/DataSets/Lenslet_8x8_Gscale/',
                        help='Direcory with training pngs')

    parser.add_argument('--test-path', type=str, default='/scratch/DataSets/Lenslet_8x8_Gscale',
                        help='Direcory with test pngs')


    parser.add_argument('--save-test',  dest='save_test', action='store_true', help='Save the predicted LFS on the test/validation')

    parser.add_argument('--std-path', type=str, default='/scratch',
                        help='Direcory to be root of all others (saved_lfs, saved_models, etc)')


    parser.add_argument('--context-size', type=int, default=64,
                        help='Size of the context [16,32,64, 128] (default 64x64))')
    parser.add_argument('--predictor-size', type=int, default=32,
                        help='Size of the predictor [8,16, 32] (default 32x32)')

    parser.add_argument('--epochs', type=int, default=100, help='Epochs to test (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64). For crop dataloaders, teh actual BS is multiplied by crops_per_image')
    parser.add_argument('--crop-mode', type=str, default="randomCrops",
                        help='crop mode to determine how the image will be cropped by the dataloader [sequential, randomCrops]')

    parser.add_argument('--loss', type=str, default='satd', help='Loss functionto minimize [mse|satd|dct]')
    parser.add_argument('--quantize-loss', dest='quantization', action='store_true', help='perform quantization during customLoss')
    parser.add_argument('--denormalize-loss', dest='denormalize_loss', action='store_true', help='denormalize block before calc. loss')
 
    parser.add_argument('--loss-mode', type=str, default='predOnly', help='Defines context for loss [predOnly|fullContext]')
    #parser.add_argument('--context-mode', type=str, default='black', help='Defines context for prediction [black|average]')

    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate (default: 0.0001)')
    parser.add_argument('--lr-gamma', type=float, default=0.2, help='Learning rate decay factor (default: 0.2)')
    parser.add_argument('--lr-min', type=float, default=0.00001, help='Learning rate decay factor (default: 0.1)')
    parser.add_argument("--lr-step-size", default=3, type=int, 
    help="UNUSED SO FAR-- Decrease lr every step-size for StepLRscheduler (non implemented) epochs ")
    parser.add_argument("--lr-scheduler", default="custom", type=str,
                        help="the lr scheduler [lr|custom]")
    parser.add_argument("--optimizer", default="adam", type=str,
                        help="the optimizer (default: adam) [sgd]")


    # parameters of the DataSet
    parser.add_argument("--dataset-name",   default='EPFL', type=str, help="Name of the dataset. (For loggin purpouses only. So far)")
    parser.add_argument("--num-views-ver",  default=1,      type=int, help="Num Views Vertical")
    parser.add_argument("--num-views-hor",  default=1,      type=int, help="Num Views Horizontally")
    parser.add_argument("--resol_ver",      default=3456,   type=int, help="Vertical Resolution")
    parser.add_argument("--resol_hor",      default=4960,   type=int, help="Horizontal Resolution")
    parser.add_argument("--bit-depth",      default=8,      type=int, help="Bit Depth")
    parser.add_argument("--limit-train",    default=-1,     type=int, help="Max num of LFs to train. (FOR QUICK TEST PURPOUSES ONLY)")
    parser.add_argument("--limit-val",      default=-1,     type=int, help="Max num of LFs to val. (FOR QUICK TEST PURPOUSES ONLY)")
    parser.add_argument('--transforms',     type=str,       default='3', help='Direcory with test pngs')

    parser.add_argument('--no-wandb', dest='wandb_active', action='store_false')

    parser.add_argument('--save', default='../runs/exp', type=str,
                        help='Output dir')
    parser.add_argument('--project-name', default='delete', type=str)
    parser.add_argument('--run-name', default='test_dump', type=str)
    #removed for redundant usage
    #parser.add_argument('--output-path', default='test_dump', type=str, metavar='PATH',
     #                   help='Path to save the Models and output LFs in folders saved_models and saved_LFs')

    #@TODO automatizar resuming simulations
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume-epoch', default=1, type=int, metavar='PATH',
                        help='The number of epochs the param was previously trained')

    
    parser.add_argument('--save-train', dest='save_train', action='store_true')


    parser.add_argument('--model', default='3k', type=str)
    parser.add_argument('--num-filters', default=32, type=int)
    parser.add_argument('--skip-connections',  default='noSkip', type=str)

    
    parser.add_argument('--prune', default=None, type=str)
    parser.add_argument('--target-sparsity',  default=0.90, type=float)
    parser.add_argument('--prune-step',  default=0.0, type=float)




    args = parser.parse_args()
    return args