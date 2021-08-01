from train_utils import *
import torch
from termcolor import colored
from loss import IAML_loss, CMD_loss, G_loss, D_loss
from network import average_view_pooling

def sketch_pretrainer(self):
    for i, data in enumerate(self.train_loader, 0):
        self.total_iter_count += 1
        # Data Load
        sketches = data[0].to(self.device) if torch.cuda.is_available() else data[0]
        cls_sketch = data[1].to(self.device) if torch.cuda.is_available() else data[1]
        sketches = torch.squeeze(sketches)
        cls_sketch = torch.squeeze(cls_sketch)
        
        ### Update Sketch CNN & Metric network ###
        self.sketch_cnn.zero_grad()
        self.sketch_metric.zero_grad()

        s_cnn_features = self.sketch_cnn(sketches)
        s_metric_features = self.sketch_metric(s_cnn_features)

        iaml_loss_sketch = IAML_loss(s_metric_features, s_metric_features, cls_sketch)
        iaml_loss_sketch.backward()
        self.sketch_optim.step()

        self.writer.add_scalar("Loss/Sketch_iaml_pre", iaml_loss_sketch, self.total_iter_count)
        if self.total_iter_count % 100 == 0:
            print("=====================================================")
            print(colored("Pre-train Sketch network step... Iteration Check: {}".format(self.total_iter_count),"blue"))
            print("Sketch Loss: {}".format(iaml_loss_sketch))

        if self.total_iter_count % 5000 == 0:
            print(colored("Save Pre-train Sketch network at {} Iteration".format(self.total_iter_count), "red"))
            trained_models = {"sketch_cnn": self.sketch_cnn, "sketch_metric": self.sketch_metric, "sketch_optim": self.sketch_optim}
            save_ckpt(self.epoch_count, self.total_iter_count, pretraining=True, models=trained_models, mode="sketch")

def model_pretrainer(self):
    for i, data in enumerate(self.train_loader, 0):
        self.total_iter_count += 1

        # Data Load
        rendered_models = data[2].to(self.device) if torch.cuda.is_available() else data[2]
        cls_model = data[3].to(self.device) if torch.cuda.is_available() else data[3]
        rendered_models = torch.squeeze(rendered_models)
        cls_model = torch.squeeze(cls_model)

        ### Update Model CNN & Metric network ###
        self.model_cnn.zero_grad()
        self.model_metric.zero_grad()

        decide_expand_dim = True
        view_num = rendered_models.shape[1]
        for i in range(view_num):
            m_cnn_feature = self.model_cnn(rendered_models[ : , i, ... ])
            if decide_expand_dim:
                m_cnn_features_sub = torch.unsqueeze(m_cnn_feature, 1)
                decide_expand_dim = False
            else:
                m_cnn_feature = torch.unsqueeze(m_cnn_feature, 1)
                m_cnn_features_sub = torch.cat((m_cnn_features_sub, m_cnn_feature), 1)
        m_cnn_features = average_view_pooling(m_cnn_features_sub)
        m_metric_features = self.model_metric(m_cnn_features)

        iaml_loss_model = IAML_loss(m_metric_features, m_metric_features, cls_model)
        iaml_loss_model.backward()
        
        self.model_optim.step()

        self.writer.add_scalar("Loss/Model_iaml_pre", iaml_loss_model, self.total_iter_count)

        if self.total_iter_count % 100 == 0:
            print("=====================================================")    
            print(colored("Pre-train Model network step... Iteration Check: {}".format(self.total_iter_count),"blue"))
            print("Model Loss: {}".format(iaml_loss_model))

        if self.total_iter_count % 5000 == 0:
            print(colored("Save Pre-train Model network at {} Iteration".format(self.total_iter_count),"red"))
            trained_models = {"model_cnn": self.model_cnn, "model_metric": self.model_metric, "model_optim": self.model_optim}
            save_ckpt(self.epoch_count, self.total_iter_count, pretraining=True, models=trained_models, mode="model")

def trans_pretrainer(self):
    for i, data in enumerate(self.train_loader, 0):
        self.total_iter_count += 1
        # Data Load
        sketches = data[0].to(self.device) if torch.cuda.is_available() else data[0]
        cls_sketch = data[1].to(self.device) if torch.cuda.is_available() else data[1]
        rendered_models = data[2].to(self.device) if torch.cuda.is_available() else data[2]
        cls_model = data[3].to(self.device) if torch.cuda.is_available() else data[3]

        sketches = torch.squeeze(sketches)
        cls_sketch = torch.squeeze(cls_sketch)
        rendered_models = torch.squeeze(rendered_models)
        cls_model = torch.squeeze(cls_model)

        ## Gradient Initialization
        self.sketch_cnn.zero_grad()
        self.sketch_metric.zero_grad()

        self.model_cnn.zero_grad()
        self.model_metric.zero_grad()

        self.transform_net.zero_grad()
        self.discriminator.zero_grad()

        # Sketch network forward
        s_cnn_features = self.sketch_cnn(sketches)
        s_metric_features = self.sketch_metric(s_cnn_features)

        # CAD Model network forward
        decide_expand_dim = True
        view_num = rendered_models.shape[1]
        for i in range(view_num):
            m_cnn_feature = self.model_cnn(rendered_models[ : , i, ... ])
            if decide_expand_dim:
                m_cnn_features_sub = torch.unsqueeze(m_cnn_feature, 1)
                decide_expand_dim = False
            else:
                m_cnn_feature = torch.unsqueeze(m_cnn_feature, 1)
                m_cnn_features_sub = torch.cat((m_cnn_features_sub, m_cnn_feature), 1)
        m_cnn_features = average_view_pooling(m_cnn_features_sub)
        m_metric_features = self.model_metric(m_cnn_features)
        
        # Transformation network forward
        trans_features = self.transform_net(s_metric_features)

        # Discriminator network forward
        trans_disc = self.discriminator(trans_features)
        model_disc = self.discriminator(m_metric_features)

        # Loss
        trans_loss = IAML_loss(trans_features, trans_features, cls_sketch) + \
            (G_loss(trans_disc) + CMD_loss(trans_features, m_metric_features, cls_sketch, cls_model))
        disc_loss = D_loss(model_disc, trans_disc)

        trans_disc_loss = (trans_loss + disc_loss)/2
        trans_disc_loss.backward()

        self.trans_optim.step()

        self.writer.add_scalar("Loss/Trans_trans_pre", trans_loss, self.total_iter_count)
        self.writer.add_scalar("Loss/Trans_disc_pre", disc_loss, self.total_iter_count)
        self.writer.add_scalar("Loss/Trans_trans_disc_pre", trans_disc_loss, self.total_iter_count)

        if self.total_iter_count % 100 == 0:
            print("=====================================================")    
            print("Pre-train Transformation network step... Iteration Check: {}".format(self.total_iter_count))
            print("Trans loss: {}, Disc loss: {}, Sum of both: {}".format(trans_loss, disc_loss, trans_disc_loss))

        if self.total_iter_count % 5000 == 0:
            print("Save Pre-train Transformation network at {} Iteration".format(self.total_iter_count))
            trained_models = {"transform_net": self.transform_net, "trans_optim": self.trans_optim}
            save_ckpt(self.epoch_count, self.total_iter_count, pretraining=True, models=trained_models, mode="trans")

def whole_trainer(self):
    for i, data in enumerate(self.train_loader, 0):
        self.total_iter_count += 1

        # Data Load
        sketches = data[0].to(self.device) if torch.cuda.is_available() else data[0]
        cls_sketch = data[1].to(self.device) if torch.cuda.is_available() else data[1]
        rendered_models = data[2].to(self.device) if torch.cuda.is_available() else data[2]
        cls_model = data[3].to(self.device) if torch.cuda.is_available() else data[3]

        sketches = torch.squeeze(sketches)
        cls_sketch = torch.squeeze(cls_sketch)
        rendered_models = torch.squeeze(rendered_models)
        cls_model = torch.squeeze(cls_model)

        ## Sketch Network update
        self.sketch_cnn.zero_grad()
        self.sketch_metric.zero_grad()
        self.model_cnn.zero_grad()
        self.model_metric.zero_grad()
        self.transform_net.zero_grad()
        self.discriminator.zero_grad()

        s_cnn_features = self.sketch_cnn(sketches)
        s_metric_features = self.sketch_metric(s_cnn_features)
        iaml_loss_sketch = IAML_loss(s_metric_features,s_metric_features,cls_sketch)
        iaml_loss_sketch.backward(retain_graph=True)
        self.sketch_optim.step()

        decide_expand_dim = True
        view_num = rendered_models.shape[1]
        for i in range(view_num):
            m_cnn_feature = self.model_cnn(rendered_models[ : , i, ... ])
            if decide_expand_dim:
                m_cnn_features_sub = torch.unsqueeze(m_cnn_feature, 1)
                decide_expand_dim = False
            else:
                m_cnn_feature = torch.unsqueeze(m_cnn_feature, 1)
                m_cnn_features_sub = torch.cat((m_cnn_features_sub, m_cnn_feature), 1)
        m_cnn_features = average_view_pooling(m_cnn_features_sub)
        m_metric_features = self.model_metric(m_cnn_features)
        iaml_loss_model = IAML_loss(m_metric_features,m_metric_features,cls_model)
        iaml_loss_model.backward(retain_graph=True)
        self.model_optim.step()

        trans_features = self.transform_net(s_metric_features)
        trans_disc = self.discriminator(trans_features)
        model_disc = self.discriminator(m_metric_features)

        trans_loss = IAML_loss(trans_features, trans_features, cls_sketch) + \
            (G_loss(trans_disc) + CMD_loss(trans_features, m_metric_features, cls_sketch, cls_model))
        disc_loss = D_loss(model_disc, trans_disc)

        trans_loss.backward(retain_graph=True, inputs=list(transform_net.parameters()))
        self.trans_optim.step()

        disc_loss.backward(inputs=list(discriminator.parameters()))
        self.disc_optim.step()

        self.writer.add_scalar("Loss/Sketch_loss", iaml_loss_sketch, self.total_iter_count)
        self.writer.add_scalar("Loss/Model_loss", iaml_loss_model, self.total_iter_count)
        self.writer.add_scalar("Loss/Trans_loss", trans_loss, self.total_iter_count)
        self.writer.add_scalar("Loss/Disc_loss", disc_loss, self.total_iter_count)

        if self.total_iter_count % 100 == 0:
            print("Total Iterative Network... Iteration Check: {}".format(self.total_iter_count))
            print("Sketch loss: {}, Model loss: {}".format(iaml_loss_sketch, iaml_loss_model))
            print("Trans loss: {}, Disc loss: {}, Sum of both: {}".format(trans_loss, disc_loss, (trans_loss+disc_loss)/2))

        if self.total_iter_count % 5000 == 0:
            print("Save Networks parameters at {} Iteration".format(self.total_iter_count))
            trained_models = {"sketch_cnn": self.sketch_cnn, "sketch_metric": self.sketch_metric, "sketch_optim": self.sketch_optim,\
                                "model_cnn": self.model_cnn, "model_metric": self.model_metric, "model_optim": self.model_optim,\
                                "transform_net": self.transform_net, "trans_optim": self.trans_optim}
            save_ckpt(self.epoch_count, self.total_iter_count, pretraining=False, models=trained_models)
            



