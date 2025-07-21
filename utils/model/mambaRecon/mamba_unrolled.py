

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import trunc_normal_

from torch.nn.modules.utils import _pair
from scipy import ndimage

from .layers.patch import PatchEmbed2D, PatchMerging2D, PatchExpand, FinalPatchExpand_X4
from .layers.data_consistency import DataConsistency
from .mamba_block import VSSLayer



logger = logging.getLogger(__name__)

class VSSM_unrolled(nn.Module): 
    def __init__(self, patch_size=4, in_chans=2, num_classes=2, depths=[2, 2, 2, 2, 2, 2], 
                 dims=[128, 128, 128, 128], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.num_features_up = int(dims[0] * 2)
        self.dims = dims
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim = 128,
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample= None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            self.layers.append(DataConsistency(128, patchify=True, patch_size=patch_size))

        self.last_dc = DataConsistency(128, patchify=False, patch_size=patch_size)
        self.norm = norm_layer(self.num_features)


        self.apply(self._init_weights)



    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    #Encoder and Bottleneck
    def forward_features(self, x, us_im, us_mask, coil_map):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, us_im, us_mask, coil_map)

        x = self.norm(x)  # B H W C

        return x


    def forward(self, x, us_mask, coil_map):
        us_im = x.clone()
        x = self.forward_features(x, us_im, us_mask, coil_map)
        x = self.last_dc(x, us_im, us_mask, coil_map)
        return x





class MambaUnrolled(nn.Module):
    # def __init__(self, config, patch_size=2, num_classes=2, zero_head=False, vis=False):
    def __init__(
        self,
        patch_size: int = 2,
        in_chans: int = 2,
        num_classes: int = 2,
        depths: list = None,
        dims: list = None,
        d_state: int = 16,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        zero_head: bool = False,
        vis: bool = False,
    ):
        super(MambaUnrolled, self).__init__()
        # self.num_classes = num_classes
        # self.zero_head = zero_head
        # self.config = config

        # self.mamba_unet =  VSSM_unrolled(
        #                         patch_size=patch_size,
        #                         num_classes=self.num_classes,
        #                         mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        #                         drop_rate=config.MODEL.DROP_RATE,
        #                         drop_path_rate=config.MODEL.DROP_PATH_RATE,
        #                         patch_norm=config.MODEL.SWIN.PATCH_NORM,
        #                         use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        # 기본값 설정
        depths = depths if depths is not None else [2, 2, 2, 2, 2, 2]
        dims   = dims   if dims   is not None else [128, 128, 128, 128]

        self.num_classes = num_classes
        self.zero_head   = zero_head

        self.mamba_unet = VSSM_unrolled(
            patch_size        = patch_size,
            in_chans          = in_chans,
            num_classes       = num_classes,
            depths            = depths,
            dims              = dims,
            d_state           = d_state,
            drop_rate         = drop_rate,
            attn_drop_rate    = attn_drop_rate,
            drop_path_rate    = drop_path_rate,
            patch_norm        = patch_norm,
            use_checkpoint    = use_checkpoint,
        )
    def forward(self, x, us_mask, coil_map):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        logits = self.mamba_unet(x, us_mask, coil_map)
        return logits