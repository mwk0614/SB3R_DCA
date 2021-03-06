import argparse

def str2bool(v): 
    if isinstance(v, bool):
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--pretraining", type=str2bool, default=True)
    parser.add_argument("--pretraining_mode", type=str, default="1234")

    # Load checkpoints from certain global step
    parser.add_argument("--sketch_global_step", type=int, default=0, help="Set as not 0 if you want to load pretrained or intermediate sketch ckpt")
    parser.add_argument("--model_global_step", type=int, default=0, help="Set as not 0 if you want to load pretrained or intermediate model ckpt")
    parser.add_argument("--trans_global_step", type=int, default=0, help="Set as not 0 if you want to load pretrained or intermediate trans ckpt")
    parser.add_argument("--whole_global_step", type=int, default=0, help="Set as not 0 if you want to load intermediate whole ckpt")

    # Data Path
    parser.add_argument("--sketch_train_dir", type=str, default="./SHREC2013/TRAINING_SKETCHES_resized/TRAINING_SKETCHES")
    parser.add_argument("--sketch_test_dir", type=str, default="./SHREC2013/TESTING_SKETCHES")
    parser.add_argument("--model_dir", type=str, default="./SHREC2013/TARGET_MODELS/rendered_models", help="not CAD itself, but rendered images from CAD")
    parser.add_argument("--model_cla_file", type=str, default="./SHREC2013_CLA_files/SHREC13_SBR_Model.cla")

    # Checkpoint directory
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--sketch_pretrained_ckpt_dir", type=str, default="sketch_pretrained")
    parser.add_argument("--model_pretrained_ckpt_dir", type=str, default="model_pretrained")
    parser.add_argument("--trans_pretrained_ckpt_dir", type=str, default="trans_pretrained")
    parser.add_argument("--whole_ckpt_dir", type=str, default="networks")

    # Training Setting
    parser.add_argument("--max_iter", type=int, default=30000, help="the number of the maximal iterative")
    parser.add_argument("--max_epoch", type=int, default=50, help="the number of the maximal epoch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_step", type=int, default=10000)

    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--K_sketch", type=int, default=16)
    parser.add_argument("--K_model", type=int, default=4)

    # Hardware Setting
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    return args
