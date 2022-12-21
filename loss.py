import torch.fft as fft
import torch
from torch import nn
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn import functional as F


def mask2image(mask):
    """
    mask: Tensor, (B, 256, H, W), value range [0, 255]
    """
    mask = torch.argmax(mask, dim=1, keepdim=True)
    mask = mask.float()/255
    return mask


class CharBonnierLoss(nn.Module):
    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(self, predict, gt):
        error = torch.sqrt(torch.square(predict-gt)+self.eps).mean()
        return error

    def __repr__(self):
        return f'CharBonnierLoss, eps:{self.eps}'


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, soft=0.1, kernel=5):
        super(SoftTargetCrossEntropy, self).__init__()
        self.soft = soft
        self.kernel = kernel
        p = (kernel-1)//2
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=p, bias=False)
        self.conv.weight = self.cal_weight()
    
    def cal_weight(self):
        w = torch.randn((1, 1, self.kernel), requires_grad=False)
        c = (self.kernel-1)//2
        w[0, 0, c] = 1 - self.soft
        v = self.soft
        for i in range(1, c+1):
            v = v*0.8
            w[0, 0, c-i] = v
            w[0, 0, c+i] = v
        w = nn.Parameter(w, requires_grad=False)
        return w

    def forward(self, predict: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # batch = gt.shape[0]
        gt = (gt*255).long().squeeze(dim=1).view(-1) # batch*W*H
        gt = F.one_hot(gt, num_classes=256).unsqueeze(dim=1).float()
        gt = self.conv(gt).squeeze(dim=1)
        predict = predict.permute(0, 2, 3, 1).view(-1, 256)
        ls = -F.log_softmax(predict, dim=-1)
        loss = torch.sum(gt * ls, dim=-1)
        return loss.mean()


class CrossEntropy(nn.Module):
    def __init__(self, smoothing=0):
        super().__init__()
        self.cri = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, predict, gt):
        """
        predict: shape=(b, 256, h, w)
        gt: shape=(b, 1, h, w), gt value range [0, 1]
        """
        gt = (gt*255).long().squeeze(dim=1)
        loss = self.cri(predict, gt)
        return loss


class PhaseLoss(nn.Module):
    def __init__(self, sim='cosin'):
        super().__init__()
        self.sim = sim
    
    def forward(self, predict, gt):
        predict_fft = fft.fftn(predict, dim=[-2, -1])
        gt_fft = fft.fftn(gt, dim=[-2, -1])
        predict_angle = torch.angle(predict_fft)/3.14159
        gt_angle = torch.angle(gt_fft)/3.14159
        predict_phase = torch.cat([torch.cos(predict_angle), torch.sin(predict_angle)], dim=1)
        gt_phase = torch.cat([torch.cos(gt_angle), torch.sin(gt_angle)], dim=1)
        if self.sim == 'cosin':
            map = torch.cosine_similarity(predict_phase, gt_phase, dim=1)
        else:
            raise NotImplementedError(f'{self.sim} is not implemented...')
        return 1 - map.mean()

    def __repr__(self):
        return f'PhaseLoss, similarity function: {self.sim}'


if __name__ == '__main__':
    print(torch.__version__)
    x = torch.randn((1, 256, 128, 128))
    x = x - x
    y = torch.randint(0, 256, (1, 128, 128)).float()/255
    for i in range(128):
        for j in range(128):
            v = y[:, i, j]
            v = (v*255).long()
            # print(v)
            x[:, v, i, j] = 100
    ce = CrossEntropy(0)
    ce = SoftTargetCrossEntropy(0.001, kernel=5)
    l = ce(x, y)
    img = mask2image(x)
    print((img-y).sum())
    print(l)
 