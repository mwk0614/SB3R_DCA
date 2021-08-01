from args import make_args
import os
import torch

args = make_args()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    # return None

def save_ckpt(epoch, iter, pretraining=False, models={}, mode="sketch"):
    # mode could be "sketch","model","trans"
    # Check if saved path exists
    if pretraining:
        if mode == "sketch":
            sketch_cnn_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.sketch_pretrained_ckpt_dir + "/" + "{}_{}".format(str(epoch), str(iter))
            sketch_metric_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.sketch_pretrained_ckpt_dir + "/" + "{}_{}".format(str(epoch), str(iter))
            sketch_optim_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.sketch_pretrained_ckpt_dir + "/" + "{}_{}".format(str(epoch), str(iter))

            check_path(sketch_cnn_ckpt_path)
            check_path(sketch_metric_ckpt_path)
            check_path(sketch_optim_ckpt_path)

            torch.save(models["sketch_cnn"], sketch_cnn_ckpt_path + "/sketch_cnn_ckpt.pth")
            torch.save(models["sketch_metric"], sketch_metric_ckpt_path + "/sketch_metric_ckpt.pth")
            torch.save(models["sketch_optim"], sketch_optim_ckpt_path + "/sketch_optim_ckpt.pth")

        if mode == "model":
            model_cnn_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.model_pretrained_ckpt_dir + "/" + "{}_{}".format(str(epoch), str(iter))
            model_metric_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.model_pretrained_ckpt_dir + "/" + "{}_{}".format(str(epoch), str(iter))
            model_optim_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.model_pretrained_ckpt_dir + "/" + "{}_{}".format(str(epoch), str(iter))

            check_path(model_cnn_ckpt_path)
            check_path(model_metric_ckpt_path)
            check_path(model_optim_ckpt_path)

            torch.save(models["model_cnn"], model_cnn_ckpt_path + "/model_cnn_ckpt.pth")
            torch.save(models["model_metric"], model_metric_ckpt_path + "/model_metric_ckpt.pth")
            torch.save(models["model_optim"], model_optim_ckpt_path + "/model_optim_ckpt.pth")

        if mode == "trans":
            transform_net_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.trans_pretrained_ckpt_dir + "/" + "{}_{}".format(str(epoch), str(iter))
            trans_optim_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.trans_pretrained_ckpt_dir + "/" + "{}_{}".format(str(epoch), str(iter))

            check_path(transform_net_ckpt_path)
            check_path(trans_optim_ckpt_path)

            torch.save(models["transform_net"], transform_net_ckpt_path + "/transform_net_ckpt.pth")
            torch.save(models["trans_optim"], trans_optim_ckpt_path + "/trans_optim_ckpt.pth")

    else:
        sketch_cnn_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.networks + "/" + "{}_{}".format(str(epoch), str(iter))
        sketch_metric_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.networks + "/" + "{}_{}".format(str(epoch), str(iter))
        sketch_optim_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.networks + "/" + "{}_{}".format(str(epoch), str(iter)) 

        model_cnn_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.networks + "/" + "{}_{}".format(str(epoch), str(iter)) 
        model_metric_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.networks + "/" + "{}_{}".format(str(epoch), str(iter))
        model_optim_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.networks + "/" + "{}_{}".format(str(epoch), str(iter)) 

        transform_net_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.networks + "/" + "{}_{}".format(str(epoch), str(iter))
        trans_optim_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.networks + "/" + "{}_{}".format(str(epoch), str(iter))

        check_path(sketch_cnn_ckpt_path)
        check_path(sketch_metric_ckpt_path)
        check_path(sketch_optim_ckpt_path)

        check_path(model_cnn_ckpt_path)
        check_path(model_metric_ckpt_path)
        check_path(model_optim_ckpt_path)

        check_path(transform_net_ckpt_path)
        check_path(trans_optim_ckpt_path)

        torch.save(models["sketch_cnn"], sketch_cnn_ckpt_path + "/sketch_cnn_ckpt.pth")
        torch.save(models["sketch_metric"], sketch_metric_ckpt_path + "/sketch_metric_ckpt.pth")
        torch.save(models["sketch_optim"], sketch_optim_ckpt_path + "/sketch_optim_ckpt.pth")

        torch.save(models["model_cnn"], model_cnn_ckpt_path + "/model_cnn_ckpt.pth")
        torch.save(models["model_metric"], model_metric_ckpt_path + "/model_metric_ckpt.pth")
        torch.save(models["model_optim"], model_optim_ckpt_path + "/model_optim_ckpt.pth")
        
        torch.save(models["transform_net"], transform_net_ckpt_path + "/transform_net_ckpt.pth")
        torch.save(models["trans_optim"], trans_optim_ckpt_path + "/trans_optim_ckpt.pth")