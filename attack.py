import numpy as np
import tensorflow as tf
from ares.loss import CrossEntropyLoss
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph


class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        self.model = model              # 网络模型
        self.batch_size = batch_size    # batch_size
        self._session = session         # 计算图
        self.dataset = dataset
        if self.dataset == 'imagenet':
            self.output_size = 1000
        else:
            self.output_size = 10
        self.xs_ph = tf.placeholder(tf.float32, shape=[None, *self.model.x_shape])     # placeholder 开辟空间
        self.ys_ph = tf.placeholder(tf.int32)     # placeholder 开辟空间
        self.labels_op = self.model.labels(self.xs_ph)                    # 获取输入图片的预测类别
        self.logits, _ = model.logits_and_labels(self.xs_ph)              # 获取网络模型的logits(pre_softmax)
        label_mask = tf.one_hot(self.ys_ph, self.output_size, on_value=1.0, off_value=0.0, dtype=tf.float32)
        correct_logit = tf.reduce_sum(label_mask * self.logits, axis=1)   # 正确的logits
        wrong_logit = tf.reduce_max((1 - label_mask) * self.logits - 1e4 * label_mask, axis=1)  # 错误的logits
        self.loss = wrong_logit - correct_logit              # loss 好 借鉴
        self.grad = tf.gradients(self.loss, self.xs_ph)[0]   # 梯度  self.loss
        ce_loss = CrossEntropyLoss(self.model)  # 交叉熵函数（Imagenet涨分）
        yan_loss = ce_loss(self.xs_ph, self.ys_ph)  # 损失
        if self.dataset == 'imagenet':
            self.yan_grad = tf.gradients(yan_loss,self.xs_ph)[0]
        else:
            self.yan_grad = tf.gradients(wrong_logit,self.xs_ph)[0]

    # 根据外部配置函数，更新攻击参数
    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6    # eps
            if self.dataset == 'imagenet':
                self.restart_num = 5        # 重启次数        7
                self.iteration = 20         # 正常迭代次数    14
                self.momentum = 1.0         # 动量法 1.0
                self.odi_num = 1
                self.alpha = self.eps/1.0   # 
                self.interval = 10           # 6最佳
                self.Sfactor = 0.6          # 0.3
            else:
                self.restart_num = 6        # 重启次数      6
                self.iteration = 16         # 正常迭代次数  16
                self.momentum = 0.0           # 动量法
                self.odi_num = 4
                self.alpha = self.eps/1.0   # 
                self.interval = 10           # 8
                self.Sfactor = 0.2          # 

    # 批量攻击操作
    def batch_attack(self, xs, ys=None, ys_target=None):
        # 设置重启次数
        best_adv = xs       # 最佳的对抗样本（odi选择的结果）
        orignal_idx = np.array(range(xs.shape[0]))
        correct_labels = self._session.run(self.labels_op, feed_dict={self.xs_ph: xs})  # 获取正确的label
        for t in range(self.restart_num):
            xs_adv = xs     # 不加噪声
            self.batch_size = xs_adv.shape[0]
            xs_lo, xs_hi = xs - self.eps, xs + self.eps  # 图片可用的扰动范围
            # odi生成初始样本
            rand_direct = np.random.uniform(-1.0, 1.0, (self.batch_size, self.output_size))  # 随机均匀分布
            loss_random = tf.tensordot(self.logits, rand_direct.astype(np.float32), axes=[[0, 1], [0, 1]])
            self.grad_ODI = tf.gradients(loss_random, self.xs_ph)[0]       # odi梯度-correct_logit
            for _ in range(self.odi_num):
                grad = self._session.run(self.grad_ODI, feed_dict={self.xs_ph: xs_adv, self.ys_ph:ys})
                grad = grad.reshape(self.batch_size, *self.model.x_shape)  # 梯度形状的规整
                grad_sign = np.sign(grad)                # 梯度符号
                xs_adv = np.clip(xs_adv + self.alpha*grad_sign, xs_lo, xs_hi)       # 对抗样本的更新(扩大2倍)
                xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)        # 图像像素范围的裁剪
            batch_shape = [self.batch_size, *self.model.x_shape]
            Ygrad = np.zeros(shape = batch_shape)
            yan_step = self.alpha
            for i in range(self.iteration):
                if (i+1)%self.interval ==0:
                    yan_step = self.alpha*self.Sfactor        # 0.5 0.3
                if (i<2):
                    noise = self._session.run(self.yan_grad, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})  # 计算梯度
                else:
                    noise = self._session.run(self.grad, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})  # 计算梯度
                noise = noise.reshape(self.batch_size, *self.model.x_shape)                            # 梯度形状的规整
                temp_noise = noise.reshape(self.batch_size,-1)
                temp_mean = np.mean(np.abs(temp_noise),axis=1)+ 1e-4
                noise = noise/ temp_mean.reshape(self.batch_size,1,1,1)
                Ygrad = self.momentum*Ygrad+noise
                grad_sign = np.sign(Ygrad)                                        # 梯度符号
                xs_adv = np.clip(xs_adv + yan_step * grad_sign, xs_lo, xs_hi)     # 对抗样本的更新  self.alpha
                xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)      # 图像像素范围的裁剪
            new_labels = self._session.run(self.labels_op, feed_dict={self.xs_ph: xs_adv})  # 对抗样本的label
            change_idx = correct_labels != new_labels
            rm_idx = orignal_idx[change_idx]              # 攻击成功的idx
            best_adv[rm_idx, :] = xs_adv[change_idx, :]   # 更新对抗样本
            idx = ~change_idx                       # 未攻击成功的idx
            xs = xs[idx]                            # 更新x
            ys = ys[idx]                            # 更新y
            correct_labels = correct_labels[idx]    # 更新真实label标签
            orignal_idx = orignal_idx[idx]          # 原始图片id
        return best_adv
