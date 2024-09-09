# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MTFViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # self.patch_norm = norm_layer(embed_dim)
        # self.patch_pred = nn.Linear(self.embed_dim, 2)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        self.cls_norm = norm_layer(embed_dim)
        self.norm_pix_loss = norm_pix_loss
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, 2)
        
        
        # self.out_indices = [5,11]
        self.out_indices = [3,7,11]
        # self.out_indices = [2,5,8,11]
        # self.out_indices = [1,3,5,7,9,11]
        # self.out_indices = [0,1,2,3,4,5,6,7,8,9,10,11]

        proj_layers = [
            torch.nn.Linear(embed_dim, embed_dim)
            for _ in range(len(self.out_indices) - 1)
        ]
        self.proj_layers = torch.nn.ModuleList(proj_layers)
        self.proj_attn = torch.nn.Linear(embed_dim, 1)
        # can be further changed
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        fused = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                if i != self.out_indices[-1]:
                    proj_x = self.proj_layers[self.out_indices.index(i)](x)
                else:
                    proj_x = x
                fused.append(proj_x)
        fused = torch.stack(fused, dim=1)  
        proj_weights = self.proj_attn(fused)
        proj_weights = F.softmax(proj_weights, dim=1)
        fused = fused * proj_weights
        fused = fused.sum(dim=1)
        return x, fused

    def forward_decoder(self, x, patch_labels):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # print(x.shape, patch_labels.shape)
        x_ = x[:, 1:, :]
        try:
            x_[patch_labels==1] = self.mask_token.half()
        except:
            x_[patch_labels==1] = self.mask_token
        # add pos embed
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        x = x[:, 1:, :]
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward_loss(self, target, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(target)
        # if self.norm_pix_loss:s*.5

        # loss = (pred - target) ** 2
        loss = torch.abs(pred - target)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print(loss.shape, mask.shape)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def convert_mask(self, mask):
        p = self.patch_embed.patch_size[0]
        
        h = w = mask.shape[2] // p
        mask = mask.reshape(shape=(mask.shape[0], h, p, w, p))
        mask = torch.einsum('nhpwq->nhwpq', mask)
        mask = mask.reshape(shape=(mask.shape[0], h * w, p**2))
        mask = mask.mean(2)
        zero = torch.zeros_like(mask)
        one  = torch.ones_like(mask)
        mask= torch.where(mask> 0.1, one, zero)

        return mask
    
    def forward(self, imgs, labels=None, mask=None, target=None, rf=False):
        
        x, fused = self.forward_encoder(imgs)
        # x = self.cls_norm(x)
        c = self.cls_norm(torch.mean(x[:, 1:, :], dim=1)) 
        y_pred = self.classifier(c)
        # c = self.cls_norm(x[:, 0])
        # c = x[:, 0]
        # c = self.cls_norm(c)
        y_pred = self.classifier(c) 
        
        if labels is None and mask is None:
            return y_pred
        labels[labels!=0] = 1
        ce_loss = F.cross_entropy(y_pred, labels)
        patch_labels = self.convert_mask(mask)
        
        pred = self.forward_decoder(fused, patch_labels)  # [N, L, p*p*3]
        rec_loss = self.forward_loss(target, pred, patch_labels)
        pred = self.unpatchify(pred)
        return rec_loss, ce_loss, pred, y_pred

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def mae_vit_base_patch16_dec512d2b(**kwargs):
    model = MTFViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    ckpt = torch.load('./mae_visualize_vit_base.pth')['model']
    msg = model.load_state_dict(ckpt, strict=False)
    print(msg)
    return model
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d2b






