import os
from args import make_args

from train import Train

def train(args):
    trainer_container = Train(args)
    if args.pretraining:
        if "1" in args.pretraining_mode:
            trainer_container.sketch_pretraining()
        if "2" in args.pretraining_mode:
            trainer_container.model_pretraining()
        if "3" in args.pretraining_mode:
            trainer_container.trans_pretraining()
    else:
        trainer_container.whole_trainer()

def test(args):
    raise NotImplementedError

if __name__ == "__main__":
    args = make_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("Working GPU: {}".format(args.gpu))
    print("Pretraining or not: {}".format(args.pretraining))
    print("Pretraining Mode: {}".format(args.pretraining_mode))

    train(args)

    