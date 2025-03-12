import torch
from torch import nn
import torch.nn.functional as F

class VATLoss(nn.Module):
    # K——循环次数；alpha——VAT在全部损失中的占比
    # xi——迭代扰动比例，论文恒取1e-6，苏剑林取10；eps——最终扰动比例，论文实验在minst上2最佳，苏剑林取1
    def __init__(self, model, k = 1, alpha = 1, xi = 1e-6, eps = 2, emb_name = 'word_embeddings'):
        super(VATLoss,self).__init__()
        self.model = model
        self.k = k
        self.alpha = alpha
        self.xi = xi
        self.eps = eps
        for (name, module) in self.model.named_modules():
            if emb_name in name:
                module.register_forward_hook(hook=self.hook)

    def hook(self, module, fea_in, fea_out):
        self.embed = fea_out.detach()
    
    def forward(self, attention_mask, logits):
        # 初始扰动 noise
        noise = torch.randn(self.embed.shape).to(logits.device) #正态分布
        noise = self.get_norm(noise)

        # 迭代求扰动
        for _ in range(self.k):
            noise.requires_grad_()
            new_embed = self.embed + self.xi * noise
            vat_logits = self.model(inputs_embeds=new_embed, attention_mask=attention_mask).logits
            vat_loss = self.kl(vat_logits, logits.detach())
            noise, = torch.autograd.grad(vat_loss, noise)
            noise = self.get_norm(noise)

        # 求loss
        new_embed = self.embed + self.eps * noise
        vat_logits = self.model(inputs_embeds=new_embed, attention_mask=attention_mask).logits
        # vat_loss = self.kl(vat_logits, logits) * self.alpha
        # 方式2
        vat_loss = self.kl(vat_logits, logits.detach()) * self.alpha
        # 方式3
        # vat_loss = (self.kl(vat_logits, logits.detach()) + self.kl(logits, vat_logits.detach())) * self.alpha

        return vat_loss
    
    @staticmethod
    def kl(inputs, targets):
        """
        计算kl散度
        inputs：tensor，logits
        targets：tensor，logits
        """
        return F.kl_div(F.log_softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction="batchmean")
    
    @staticmethod
    def get_norm(grad, norm_type='inf'):
        """
        L0,L1,L2正则，对于扰动计算
        """
        if norm_type == 'l2':
            direction = grad / (grad.norm(p = 2, dim=-1, keepdim=True) + 1e-8)
        elif norm_type == 'l1':
            direction = grad / (torch.sum(torch.abs(grad), dim=-1, keepdim=True) + 1e-8)
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + 1e-8)
        return direction