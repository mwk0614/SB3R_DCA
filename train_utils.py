from args import make_args
import os
import torch
import glob

args = make_args()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    # return None

def save_ckpt(self, pretraining=False, mode="sketch"):
    # mode could be "sketch","model","trans"
    # Check if saved path exists
    if pretraining:
        if mode == "sketch":
            sketch_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.sketch_pretrained_ckpt_dir
            check_path(sketch_ckpt_path)

            torch.save(self.sketch_cnn, sketch_ckpt_path + "/sketch_cnn_{}.pth".format(self.global_step))
            torch.save(self.sketch_metric, sketch_ckpt_path + "/sketch_metric_{}.pth".format(self.global_step))
            torch.save(self.sketch_optim, sketch_ckpt_path + "/sketch_optim_{}.pth".format(self.global_step))

        if mode == "model":
            model_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.model_pretrained_ckpt_dir
            check_path(model_ckpt_path)

            torch.save(self.model_cnn, model_ckpt_path + "/model_cnn_{}.pth".format(self.global_step))
            torch.save(self.model_metric, model_ckpt_path + "/model_metric_{}.pth".format(self.global_step))
            torch.save(self.model_optim, model_ckpt_path + "/model_optim_{}.pth".format(self.global_step))

        if mode == "trans":
            trans_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.trans_pretrained_ckpt_dir
            check_path(trans_ckpt_path)

            torch.save(self.transform_net, trans_ckpt_path + "/transform_net_{}.pth".format(self.global_step))
            torch.save(self.trans_optim, trans_ckpt_path + "/trans_optim_{}.pth".format(self.global_step))

    else:
        whole_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.whole_ckpt_dir
        check_path(whole_ckpt_path)

        torch.save(self.sketch_cnn, whole_ckpt_path + "/sketch_cnn_{}.pth".format(self.global_step))
        torch.save(self.sketch_metric, whole_ckpt_path + "/sketch_metric_{}.pth".format(self.global_step))
        torch.save(self.sketch_optim, whole_ckpt_path + "/sketch_optim_{}.pth".format(self.global_step))

        torch.save(self.model_cnn, whole_ckpt_path + "/model_cnn_{}.pth".format(self.global_step))
        torch.save(self.model_metric, whole_ckpt_path + "/model_metric_{}.pth".format(self.global_step))
        torch.save(self.model_optim, whole_ckpt_path + "/model_optim_{}.pth".format(self.global_step))
        
        torch.save(self.transform_net, whole_ckpt_path + "/transform_net_{}.pth".format(self.global_step))
        torch.save(self.trans_optim, whole_ckpt_path + "/trans_optim_{}.pth".format(self.global_step))

        torch.save(self.discriminator, whole_ckpt_path + "/discriminator_{}.pth".format(self.global_step))
        torch.save(self.disc_optim, whole_ckpt_path + "/disc_optim_{}.pth".format(self.global_step))



def load_ckpt(self, pretraining=False, mode="sketch"):
    if pretraining:
        if mode == "sketch":
            sketch_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.sketch_pretrained_ckpt_dir

            # if self.args.sketch_global_step == 0:
            #     sketch_ckpts_list = sorted(glob.glob(sketch_ckpt_path+"/*"))
            #     latest_step = (sketch_ckpts_list[-1].split("_")[-1]).split(".")[0]
            #     self.sketch_cnn.load_state_dict(torch.load(glob.glob(sketch_ckpt_path + "/sketch_cnn_{}.pth".format(int(latest_step)))))
            #     self.sketch_metric.load_state_dict(torch.load(sketch_ckpt_path + "/sketch_metric_{}.pth".format(int(latest_step))))
            #     self.sketch_optim.load_state_dict(torch.load(sketch_ckpt_path + "/sketch_optim_{}.pth".format(int(latest_step))))
            #     print("Load {}th sketch checkpoint".format(int(latest_step)))
            # else:
            assert self.args.sketch_global_step != 0
            self.sketch_cnn.load_state_dict(torch.load(sketch_ckpt_path + "/sketch_cnn_{}.pth".format(self.args.sketch_global_step)))
            self.sketch_metric.load_state_dict(torch.load(sketch_ckpt_path + "/sketch_metric_{}.pth".format(self.args.sketch_global_step)))
            self.sketch_optim.load_state_dict(torch.load(sketch_ckpt_path + "/sketch_optim_{}.pth".format(self.args.sketch_global_step)))
            print("Load {}th sketch checkpoint".format(self.args.sketch_global_step))

        if mode == "model":
            model_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.model_pretrained_ckpt_dir

            # if self.args.model_global_step == 0:
            #     model_ckpts_list = sorted(glob.glob(model_ckpt_path+"/*"))
            #     latest_step = (model_ckpts_list[-1].split("_")[-1]).split(".")[0]
            #     self.model_cnn.load_state_dict(torch.load(glob.glob(model_ckpt_path + "/model_cnn_{}.pth".format(int(latest_step)))))
            #     self.model_metric.load_state_dict(torch.load(model_ckpt_path + "/model_metric_{}.pth".format(int(latest_step))))
            #     self.model_optim.load_state_dict(torch.load(model_ckpt_path + "/model_optim_{}.pth".format(int(latest_step))))
            #     print("Load {}th model checkpoint".format(int(latest_step)))
            # else:
            assert self.args.model_global_step != 0
            self.model_cnn.load_state_dict(torch.load(model_ckpt_path + "/model_cnn_{}.pth".format(self.args.model_global_step)))
            self.model_metric.load_state_dict(torch.load(model_ckpt_path + "/model_metric_{}.pth".format(self.args.model_global_step)))
            self.model_optim.load_state_dict(torch.load(model_ckpt_path + "/model_optim_{}.pth".format(self.args.model_global_step)))
            print("Load {}th model checkpoint".format(self.args.model_global_step))

        if mode == "trans":
            trans_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.trans_pretrained_ckpt_dir

            # if self.args.trans_global_step == 0:
            #     trans_ckpts_list = sorted(glob.glob(trans_ckpt_path+"/*"))
            #     latest_step = (trans_ckpts_list[-1].split("_")[-1]).split(".")[0]
            #     self.transform_net.load_state_dict(torch.load(glob.glob(trans_ckpt_path + "/transform_net{}.pth".format(int(latest_step)))))
            #     self.trans_optim.load_state_dict(torch.load(trans_ckpt_path + "/trans_optim_{}.pth".format(int(latest_step))))
            #     print("Load {}th trans checkpoint".format(int(latest_step)))
            # else:
            assert self.args.trans_global_step != 0
            self.transform_net.load_state_dict(torch.load(trans_ckpt_path + "/transform_net_{}.pth".format(self.args.trans_global_step)))
            self.trans_optim.load_state_dict(torch.load(trans_ckpt_path + "/trans_optim_{}.pth".format(self.args.trans_global_step)))
            print("Load {}th trans checkpoint".format(self.args.trans_global_step))


    else:
        sketch_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.sketch_pretrained_ckpt_dir
        model_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.model_pretrained_ckpt_dir
        trans_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.trans_pretrained_ckpt_dir
        whole_ckpt_path = args.ckpt_dir + "/{}/".format(args.trials) + args.whole_ckpt_dir

        if self.args.whole_global_step == 0:
            # Load Pretraining Networks ckpt
            self.sketch_cnn.load_state_dict(torch.load(sketch_ckpt_path + "/sketch_cnn_{}.pth".format(self.args.sketch_global_step)))
            self.sketch_metric.load_state_dict(torch.load(sketch_ckpt_path + "/sketch_metric_{}.pth".format(self.args.sketch_global_step)))
            self.sketch_optim.load_state_dict(torch.load(sketch_ckpt_path + "/sketch_optim_{}.pth".format(self.args.sketch_global_step)))

            self.model_cnn.load_state_dict(torch.load(model_ckpt_path + "/model_cnn_{}.pth".format(self.args.model_global_step)))
            self.model_metric.load_state_dict(torch.load(model_ckpt_path + "/model_metric_{}.pth".format(self.args.model_global_step)))
            self.model_optim.load_state_dict(torch.load(model_ckpt_path + "/model_optim_{}.pth".format(self.args.model_global_step)))
            
            self.transform_net.load_state_dict(torch.load(trans_ckpt_path + "/transform_net_{}.pth".format(self.args.trans_global_step)))
            self.trans_optim.load_state_dict(torch.load(trans_ckpt_path + "/trans_optim_{}.pth".format(self.args.trans_global_step)))

            print("======================================")
            print("Load {}th sketch pretrained-checkpoint".format(self.args.sketch_global_step))
            print("Load {}th model pretrained-checkpoint".format(self.args.model_global_step))
            print("Load {}th trans pretrained-checkpoint".format(self.args.trans_global_step))
            print("======================================")

        else:
            self.sketch_cnn.load_state_dict(torch.load(whole_ckpt_path + "/sketch_cnn_{}.pth".format(self.args.whole_global_step)))
            self.sketch_metric.load_state_dict(torch.load(whole_ckpt_path + "/sketch_metric_{}.pth".format(self.args.whole_global_step)))
            self.sketch_optim.load_state_dict(torch.load(whole_ckpt_path + "/sketch_optim_{}.pth".format(self.args.whole_global_step)))

            self.model_cnn.load_state_dict(torch.load(whole_ckpt_path + "/model_cnn_{}.pth".format(self.args.whole_global_step)))
            self.model_metric.load_state_dict(torch.load(whole_ckpt_path + "/model_metric_{}.pth".format(self.args.whole_global_step)))
            self.model_optim.load_state_dict(torch.load(whole_ckpt_path + "/model_optim_{}.pth".format(self.args.whole_global_step)))
            
            self.transform_net.load_state_dict(torch.load(whole_ckpt_path + "/transform_net_{}.pth".format(self.args.whole_global_step)))
            self.trans_optim.load_state_dict(torch.load(whole_ckpt_path + "/trans_optim_{}.pth".format(self.args.whole_global_step)))

            self.discriminator.load_state_dict(torch.load(whole_ckpt_path + "/discriminator_{}.pth".format(self.args.whole_global_step)))
            self.disc_optim.load_state_dict(torch.load(whole_ckpt_path + "/disc_optim_{}.pth".format(self.args.whole_global_step)))
            print("Load {}th sketch checkpoint".format(self.args.whole_global_step))
            print("Load {}th model checkpoint".format(self.args.whole_global_step))
            print("Load {}th trans checkpoint".format(self.args.whole_global_step))
