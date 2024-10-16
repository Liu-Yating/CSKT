from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from timm.models.layers import drop_path
import torch
from torch import nn
from .until_module import LayerNorm, ACT2FN
from collections import OrderedDict

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class Adapter_MLP(nn.Module):
    def __init__(self, c_in, c_out, reduction=8):
        super(Adapter_MLP, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(c_in, c_in // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(c_in // reduction, c_out, bias=False),
        #     # nn.ReLU(inplace=True)
        # )
        self.down =  nn.Linear(c_in, c_in // reduction, bias=False)
        self.ac = nn.ReLU(inplace=True)
        self.up = nn.Linear(c_in // reduction, c_out, bias=False)
        
        # self.linear = nn.Linear(c_in, c_out)

    def forward(self, x):
        res = x
        x = self.ac(self.down(x)) 
        x = self.up(x)
        # out = res + x
        # return out
        return x

class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, compound_prompt_nctx=2,
                 text_layer=False, i=0):
        super().__init__()
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        # This must be consistent with the config file prompt, todo
        # self.compound_prompt_nctx = design_details['maple_length']
        self.compound_prompt_nctx = compound_prompt_nctx
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False
        
        self.scale = 0.1   
        self.ln_3 = LayerNorm(d_model)
        self.adapter_mlp = Adapter_MLP(d_model, d_model)

    def attention(self, x: torch.Tensor, attn_mask_: torch.Tensor):
        # attn_mask_ = self.attn_mask
        # if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
        #     attn_mask_ = self.attn_mask(x.size(0))  # LND

        # attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        if attn_mask_ == None:
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]
        attn_mask_ = attn_mask_.repeat_interleave(self.n_head, dim=0)
        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        inputs, attn_mask = para_tuple
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        
        if not self.text_layer:  #image
            self.scale = 0.1
            #print("image scale %s"%self.scale)
        else:                    #text
            self.scale = 4
            #print("text scale %s"%self.scale)
        
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                # This means that deeper compound prompts are turned on
                # Here it behaves differently for text and visual side
                # Forward function is same for both

                if not self.text_layer:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Remove the outputs produced by learnable tokens of previous layer
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        # Create/configure learnable tokens of this layer
                        visual_context = compound_prompts_deeper[counter]  # extract the correct index
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, by replacing previous
                        # layer learnable tokens
                        x = torch.cat([prefix, visual_context], dim=0)

                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                else:
                    
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Appending the learnable tokens in different way
                        # x -> [77, NCLS, DIM]
                        # First remove the learnable tokens from previous layer
                        prefix = x[:1, :, :]
                        suffix = x[1 + self.compound_prompt_nctx:, :, :]
                        # Create/configure learnable tokens of this layer
                        textual_context = compound_prompts_deeper[counter]
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, replaced by previous
                        # layer learnable tokens
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
        x = x + self.attention(self.ln_1(x), attn_mask)
        #x = x + self.mlp(self.ln_2(x))
        x = x + self.mlp(self.ln_2(x)) + self.scale * self.adapter_mlp(self.ln_3(x))
        return ([x, compound_prompts_deeper, counter], attn_mask)  # return again as a list, so that nn.seq can work


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, text_layer=False, design_details=None):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        compound_prompt_nctx = design_details['n_ctx'] * 2
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, compound_prompt_nctx, text_layer, i) for i in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]
    
class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, stride_size:int, width: int, layers: int, heads: int, output_dim: int,
                 design_details=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
        num_patches = self.num_x * self.num_y

        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    # def forward_no_mlm(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts, mask=None):
    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts, mask=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),   #将768张量复制为 8*1*768,广播机制
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()#expand -1，表示将该维度扩展到适应其他维度的大小
            x = torch.cat([x, visual_ctx], dim=1) # add prompt 
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Again combine the inputs, so nn.sequential can work
        # mask: the effect is slight
        # mask_ = mask.flatten().unsqueeze(-1)
        # mask_ = mask_.unsqueeze(-1).repeat(1, 52, 52)
        outputs = self.transformer([x, compound_deeper_prompts, 0], mask)  # third argument is counter
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x