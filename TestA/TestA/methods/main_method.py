import torch
import torch.nn as nn
import numpy as np
import random

from methods.gnn import GNN_nl
from methods import backbone_multiblock
from methods.tools import *
from methods.meta_template import MetaTemplate


class MainMethod(MetaTemplate):
    maml = False

    def __init__(self, model_func, n_way, n_support, tf_path=None):
        super(MainMethod, self).__init__(model_func, n_way, n_support, tf_path=tf_path)

        # 定义损失函数
        self.loss_fn = nn.CrossEntropyLoss()

        # 定义度量函数
        self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False))
        self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)

        # 定义全局分类器
        self.method = 'GnnNet'
        self.classifier = nn.Linear(self.feature.final_feat_dim, 64)

        # 固定支持集标签
        support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
        support_label = torch.zeros(self.n_way * self.n_support, self.n_way).scatter(1, support_label, 1).view(
            self.n_way, self.n_support, self.n_way)
        support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, self.n_way)], dim=1)
        self.support_label = support_label.view(1, -1, self.n_way)

    def cuda(self):
        self.feature.cuda()
        self.fc.cuda()
        self.gnn.cuda()
        self.classifier.cuda()
        self.support_label = self.support_label.cuda()
        return self

    def forward_gnn(self, zs):
        """
        通过图神经网络计算得分。

        Args:
            zs (list): 查询样本的特征列表。

        Returns:
            scores (torch.Tensor): 计算得到的得分。
        """
        # 拼接节点特征和标签
        nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)

        # 通过图神经网络计算得分
        scores = self.gnn(nodes)

        # 调整得分形状
        scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0,
                                                                                                         2).contiguous().view(
            -1, self.n_way)

        # 返回计算得到的得分
        return scores

    def set_forward_loss(self, x):
        """
        计算模型在前向传播过程中的损失。

        Args:
            x (torch.Tensor): 输入数据。

        Returns:
            scores (torch.Tensor): 计算得到的得分。
            loss (torch.Tensor): 计算得到的损失。
        """
        # 生成查询集的标签
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()

        # 计算得分
        scores = self.set_forward(x)

        # 计算损失
        loss = self.loss_fn(scores, y_query)

        # 返回得分和损失
        return scores, loss

    # 处理输入数据 x 并返回代表模型预测的分数
    def set_forward(self, x, is_feature=False):
        # 将输入张量移动到GPU（如果可用）
        x = x.cuda()

        if is_feature:
            # 如果输入是特征张量，重塑特征张量：n_way * n_s + 15 * f
            assert (x.size(1) == self.n_support + 15)
            # 将特征张量重塑为 (n_way * (n_support + 15), f)
            z = self.fc(x.view(-1, *x.size()[2:]))
            # 将特征张量重塑回 (n_way, n_support + 15, f)
            z = z.view(self.n_way, -1, z.size(1))
        else:
            # 如果输入不是特征张量，使用编码器获取特征
            # 将输入张量重塑为 (n_way * (n_support + 15), ...)
            x = x.view(-1, *x.size()[2:])
            # 通过编码器获取特征，然后通过全连接层
            z = self.fc(self.feature(x))
            # 将特征张量重塑回 (n_way, n_support + 15, f)
            z = z.view(self.n_way, -1, z.size(1))

        # 为度量函数堆叠特征：n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
        # 对于每个查询样本，将支持特征和相应的查询特征连接起来
        z_stack = [
            torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1,
                                                                                                            z.size(2))
            for i in range(self.n_query)]
        # 确保堆叠张量具有预期的形状
        assert (z_stack[0].size(1) == self.n_way * (self.n_support + 1))

        # 通过图神经网络（GNN）计算相似度分数
        scores = self.forward_gnn(z_stack)

        # 返回计算的分数
        return scores

    def set_statues_of_modules(self, flag):
        if flag == 'eval':
            self.feature.eval()
            self.fc.eval()
            self.gnn.eval()
            self.classifier.eval()
        elif flag == 'train':
            self.feature.train()
            self.fc.train()
            self.gnn.train()
            self.classifier.train()
        return

    def adversarial_attack_Incre(self, epoch, x_ori, y_ori, epsilon_list):
        x_ori = x_ori.cuda()
        y_ori = y_ori.cuda()
        x_size = x_ori.size()
        x_ori = x_ori.view(x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4])
        y_ori = y_ori.view(x_size[0] * x_size[1])

        # if not adv, set default = 'None'
        adv_phase_spectrum_block1 = 'None'
        adv_phase_spectrum_block2 = 'None'
        adv_phase_spectrum_block3 = 'None'

        # forward and set the grad = True
        blocklist = 'block123'

        if ('1' in blocklist and epsilon_list[0] != 0):
            # forward block1
            x_ori_block1 = self.feature.forward_block1(x_ori)
            feat_size_block1 = x_ori_block1.size()
            phase_spectrum_block1 = extract_phase_spectrum(x_ori_block1).cuda().requires_grad_(True)
            # set them as learnable parameters
            phase_spectrum_block1 = torch.nn.Parameter(phase_spectrum_block1)
            phase_spectrum_block1.requires_grad_(True)
            # print(f"phase_spectrum_block1.requires_grad: {phase_spectrum_block1.requires_grad}")
            # print(f"phase_spectrum_block1.is_leaf: {phase_spectrum_block1.is_leaf}")
            #  归一化处理相位扰动后的特征
            x_ori_block1 = (
                    (x_ori_block1 - x_ori_block1.mean(dim=(2, 3), keepdim=True)) / (
                    x_ori_block1.std(dim=(2, 3), keepdim=True) + 1e-5)
            )
            x_ori_block1 = x_ori_block1 * torch.cos(phase_spectrum_block1) + x_ori_block1 * torch.sin(
                phase_spectrum_block1)

            # pass the rest model
            x_ori_block2 = self.feature.forward_block2(x_ori_block1)
            x_ori_block3 = self.feature.forward_block3(x_ori_block2)
            x_ori_block4 = self.feature.forward_block4(x_ori_block3)
            x_ori_fea = self.feature.forward_rest(x_ori_block4)
            x_ori_output = self.classifier.forward(x_ori_fea)

            # calculate initial pred, loss and acc
            ori_pred = x_ori_output.max(1, keepdim=True)[1]
            ori_loss = self.loss_fn(x_ori_output, y_ori)
            ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]

            # zero all the existing gradients
            self.feature.zero_grad()
            self.classifier.zero_grad()

            # backward loss
            ori_loss.backward(retain_graph=True)

            # collect datagrad
            grad_phase_spectrum_block1 = torch.autograd.grad(
    ori_loss, phase_spectrum_block1, retain_graph=True
)[0]

            # fgsm style attack
            index = torch.randint(0, len(epsilon_list), (1,))[0]
            epsilon = epsilon_list[index]

            adv_phase_spectrum_block1 = perturb_phase_spectrum(phase_spectrum_block1, epsilon, 1, iterations=epoch,
                                                               loss_fn=self.loss_fn)

        # add zero_grad
        self.feature.zero_grad()
        self.classifier.zero_grad()

        if ('2' in blocklist and epsilon_list[1] != 0):
            # forward block1
            x_ori_block1 = self.feature.forward_block1(x_ori)
            # update adv_block1
            x_adv_block1 = change_new_phase_spectrum(x_ori_block1, adv_phase_spectrum_block1, p_thred=0)
            # forward block2
            x_ori_block2 = self.feature.forward_block2(x_adv_block1)
            # calculate phase spectrum
            phase_spectrum_block2 = extract_phase_spectrum(x_ori_block2).cuda().requires_grad_(True)
            # set them as learnable parameters
            phase_spectrum_block2.requires_grad_(True)
            # print(f"phase_spectrum_block2.requires_grad: {phase_spectrum_block2.requires_grad}")
            # print(f"phase_spectrum_block2.is_leaf: {phase_spectrum_block2.is_leaf}")

            # 归一化处理相位扰动后的特征
            x_ori_block2 = (
                    (x_ori_block2 - x_ori_block2.mean(dim=(2, 3), keepdim=True)) / (
                    x_ori_block2.std(dim=(2, 3), keepdim=True) + 1e-5)
            )
            x_ori_block2 = x_ori_block2 * torch.cos(phase_spectrum_block2) + x_ori_block2 * torch.sin(
                phase_spectrum_block2)
            # pass the rest model
            x_ori_block3 = self.feature.forward_block3(x_ori_block2)
            x_ori_block4 = self.feature.forward_block4(x_ori_block3)
            x_ori_fea = self.feature.forward_rest(x_ori_block4)
            x_ori_output = self.classifier.forward(x_ori_fea)
            # calculate initial pred, loss and acc
            ori_pred = x_ori_output.max(1, keepdim=True)[1]
            ori_loss = self.loss_fn(x_ori_output, y_ori)
            ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]
            # zero all the existing gradients
            self.feature.zero_grad()
            self.classifier.zero_grad()
            # backward loss
            ori_loss.backward(retain_graph=True)
            # collect datagrad
            grad_phase_spectrum_block2 = torch.autograd.grad(
    ori_loss, phase_spectrum_block2, retain_graph=True
)[0]
            # fgsm style attack
            index = torch.randint(0, len(epsilon_list), (1,))[0]
            epsilon = epsilon_list[index]
            adv_phase_spectrum_block2 = perturb_phase_spectrum(phase_spectrum_block2, epsilon, 10, iterations=epoch,
                                                               loss_fn=self.loss_fn)
            # print(f"phase_spectrum_block2.grad_fn: {phase_spectrum_block2.grad_fn}")
            # print(f"ori_loss.grad_fn: {ori_loss.grad_fn}")
            # print(f"x_ori_block2.grad_fn: {x_ori_block2.grad_fn}")

        # add zero_grad
        self.feature.zero_grad()
        self.classifier.zero_grad()

        if ('3' in blocklist and epsilon_list[2] != 0):
            # forward block1, block2, block3
            x_ori_block1 = self.feature.forward_block1(x_ori)
            x_adv_block1 = change_new_phase_spectrum(x_ori_block1, adv_phase_spectrum_block1, p_thred=0)
            x_ori_block2 = self.feature.forward_block2(x_adv_block1)
            x_adv_block2 = change_new_phase_spectrum(x_ori_block2, adv_phase_spectrum_block2, p_thred=0)
            x_ori_block3 = self.feature.forward_block3(x_adv_block2)
            # calculate phase spectrum
            phase_spectrum_block3 = extract_phase_spectrum(x_ori_block3).cuda().requires_grad_(True)
            # print(f"phase_spectrum_block3.requires_grad: {phase_spectrum_block3.requires_grad}")
            # print(f"phase_spectrum_block3.is_leaf: {phase_spectrum_block3.is_leaf}")
            # print(f"x_ori_block3 before phase_spectrum manipulation: {x_ori_block3.clone().detach()}")
            x_ori_block3 = (
                    (x_ori_block3 - x_ori_block3.mean(dim=(2, 3), keepdim=True)) /
                    (x_ori_block3.std(dim=(2, 3), keepdim=True) + 1e-5)
            )
            x_ori_block3 = x_ori_block3 * torch.cos(phase_spectrum_block3) + x_ori_block3 * torch.sin(
                phase_spectrum_block3)

            # pass the rest model
            x_ori_block4 = self.feature.forward_block4(x_ori_block3)
            x_ori_fea = self.feature.forward_rest(x_ori_block4)
            x_ori_output = self.classifier.forward(x_ori_fea)
            # calculate initial pred, loss and acc
            ori_pred = x_ori_output.max(1, keepdim=True)[1]
            ori_loss = self.loss_fn(x_ori_output, y_ori)
            ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]
            # zero all the existing gradients
            self.feature.zero_grad()
            self.classifier.zero_grad()
            # backward loss
            ori_loss.backward(retain_graph=True)
            # print(f"phase_spectrum_block3.requires_grad: {phase_spectrum_block3.requires_grad}")
            # print(f"x_ori_block3.grad_fn: {x_ori_block3.grad_fn}")
            # print(f"x_ori_block4.grad_fn: {x_ori_block4.grad_fn}")
            # print(f"x_ori_fea.grad_fn: {x_ori_fea.grad_fn}")
            # print(f"x_ori_output.grad_fn: {x_ori_output.grad_fn}")

            # collect datagrad
            grad_phase_spectrum_block3 = torch.autograd.grad(
    ori_loss, phase_spectrum_block3, retain_graph=True
)[0]
            # fgsm style attack
            index = torch.randint(0, len(epsilon_list), (1,))[0]
            epsilon = epsilon_list[index]
            adv_phase_spectrum_block3 = perturb_phase_spectrum(phase_spectrum_block3, epsilon, 10, iterations=epoch,
                                                               loss_fn=self.loss_fn)

        return adv_phase_spectrum_block1, adv_phase_spectrum_block2, adv_phase_spectrum_block3

    # 5. 计算损失
    def set_forward_loss_method(self, epoch,x_ori, global_y, epsilon_list):
        ##################################################################
        # 0. first cp x_adv from x_ori
        x_adv = x_ori

        ##################################################################
        # 1. styleAdv
        self.set_statues_of_modules('eval')

        adv_phase_spectrum_block1, adv_phase_spectrum_block2, adv_phase_spectrum_block3 = self.adversarial_attack_Incre(epoch=epoch,
            x_ori=x_ori, y_ori=global_y, epsilon_list=epsilon_list)

        self.feature.zero_grad()
        self.fc.zero_grad()
        self.classifier.zero_grad()
        self.gnn.zero_grad()

        #################################################################
        # 2. forward and get loss
        self.set_statues_of_modules('train')

        # define y_query for FSL
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()

        # forward x_ori
        x_ori = x_ori.cuda()
        x_size = x_ori.size()
        x_ori = x_ori.view(x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4])
        global_y = global_y.view(x_size[0] * x_size[1]).cuda()
        x_ori_block1 = self.feature.forward_block1(x_ori)
        x_ori_block2 = self.feature.forward_block2(x_ori_block1)
        x_ori_block3 = self.feature.forward_block3(x_ori_block2)
        x_ori_block4 = self.feature.forward_block4(x_ori_block3)
        x_ori_fea = self.feature.forward_rest(x_ori_block4)

        # ori cls global loss
        scores_cls_ori = self.classifier.forward(x_ori_fea)
        loss_cls_ori = self.loss_fn(scores_cls_ori, global_y)
        acc_cls_ori = (scores_cls_ori.max(1, keepdim=True)[1] == global_y).type(torch.float).sum().item() / \
                      global_y.size()[0]

        # ori FSL scores and losses
        x_ori_z = self.fc(x_ori_fea)
        x_ori_z = x_ori_z.view(self.n_way, -1, x_ori_z.size(1))
        x_ori_z_stack = [
            torch.cat([x_ori_z[:, :self.n_support], x_ori_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(
                1, -1, x_ori_z.size(2)) for i in range(self.n_query)]
        assert (x_ori_z_stack[0].size(1) == self.n_way * (self.n_support + 1))
        scores_fsl_ori = self.forward_gnn(x_ori_z_stack)
        loss_fsl_ori = self.loss_fn(scores_fsl_ori, y_query)

        # forward x_adv
        x_adv = x_adv.cuda()
        x_adv = x_adv.view(x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4])
        x_adv_block1 = self.feature.forward_block1(x_adv)

        # Apply phase spectrum perturbation
        x_adv_block1_newStyle = change_new_phase_spectrum(x_adv_block1, adv_phase_spectrum_block1, p_thred=0)
        x_adv_block2 = self.feature.forward_block2(x_adv_block1_newStyle)
        x_adv_block2_newStyle = change_new_phase_spectrum(x_adv_block2, adv_phase_spectrum_block2, p_thred=0)
        x_adv_block3 = self.feature.forward_block3(x_adv_block2_newStyle)
        x_adv_block3_newStyle = change_new_phase_spectrum(x_adv_block3, adv_phase_spectrum_block3, p_thred=0)
        x_adv_block4 = self.feature.forward_block4(x_adv_block3_newStyle)
        x_adv_fea = self.feature.forward_rest(x_adv_block4)

        # adv cls global loss
        scores_cls_adv = self.classifier.forward(x_adv_fea)
        loss_cls_adv = self.loss_fn(scores_cls_adv, global_y)
        acc_cls_adv = (scores_cls_adv.max(1, keepdim=True)[1] == global_y).type(torch.float).sum().item() / \
                      global_y.size()[0]

        # adv FSL scores and losses
        x_adv_z = self.fc(x_adv_fea)
        x_adv_z = x_adv_z.view(self.n_way, -1, x_adv_z.size(1))
        x_adv_z_stack = [
            torch.cat([x_adv_z[:, :self.n_support], x_adv_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(
                1, -1, x_adv_z.size(2)) for i in range(self.n_query)]
        assert (x_adv_z_stack[0].size(1) == self.n_way * (self.n_support + 1))
        scores_fsl_adv = self.forward_gnn(x_adv_z_stack)
        loss_fsl_adv = self.loss_fn(scores_fsl_adv, y_query)

        # print('scores_fsl_adv:', scores_fsl_adv.mean(), 'loss_fsl_adv:', loss_fsl_adv, 'scores_cls_adv:', scores_cls_adv.mean(), 'loss_cls_adv:', loss_cls_adv)
        return scores_fsl_ori, loss_fsl_ori, scores_cls_ori, loss_cls_ori, scores_fsl_adv, loss_fsl_adv, scores_cls_adv, loss_cls_adv
