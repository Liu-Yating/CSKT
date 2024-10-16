import copy
import random 
from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights, tokenize, convert_weights_prompt
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from utils.simple_tokenizer import SimpleTokenizer
import math

tokenizer = SimpleTokenizer()

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)]) 

    
class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        
        ###################################### T2V
        n_ctx = cfg.n_ctx 
        ctx_init = "a photo of a person or a pedestrian"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.compound_prompts_depth = cfg.depth 
        if ctx_init and (n_ctx) <= 8: 
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = tokenize(ctx_init)
            embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        self.tmap = nn.Linear(ctx_dim, 768)
      
        self.ctx = nn.Parameter(ctx_vectors)
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)]) 
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        
        self.n_ctx = n_ctx
        prefix = " ".join(["X"] * n_ctx) 
        ctx_init =  prefix + " " + ctx_init 
        tokenized_prompts = torch.tensor(tokenize(ctx_init))
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        
        ############################### V2T
        # random initialization
        vision_vectors = torch.empty(n_ctx, 768, dtype=dtype)
        nn.init.normal_(vision_vectors, std=0.02)
        self.vision = nn.Parameter(vision_vectors)
        
        self.compound_prompts_vision = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                                      for _ in range(self.compound_prompts_depth - 1)])   
        for single_para in self.compound_prompts_vision:
            nn.init.normal_(single_para, std=0.02)   
            
        # Also make corresponding projection layers, for each prompt
        self.vmap = nn.Linear(768, ctx_dim)
         
        single_layer_vision = nn.Linear(768, ctx_dim)
        self.compound_prompt_projections_vision = _get_clones(single_layer_vision, self.compound_prompts_depth - 1)
        

    def forward(self):
        ctx = self.ctx.half()
        vp = self.vision.half()
        visual_deep_prompts = []
        context_deep_prompts = []
                            
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index].half()))  


        for index, layer in enumerate(self.compound_prompt_projections_vision):
            context_deep_prompts.append(layer(self.compound_prompts_vision[index].half()))
            
        return ctx, self.tmap(self.ctx.half()), self.compound_prompts_text, visual_deep_prompts, \
           vp,  self.vmap(self.vision.half()), self.compound_prompts_vision, context_deep_prompts  

            

  

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg, state_dict = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size, args.n_ctx)
        self.embed_dim = base_cfg['embed_dim']

        self.prompt_learner = MultiModalPromptLearner(args, self.base_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.apply(self.init_weights) # random init must before loading pretrain
        self.base_model.load_param(state_dict)
           
        # covert model to fp16
        if torch.cuda.is_available():
            convert_weights(self.base_model)
            convert_weights_prompt(self.prompt_learner)

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        
    def init_weights(self, module):
        """ Initialize the weights.
        """
        # print(module)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
         
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']

        ctx, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, vision, shared_vision, deep_compound_prompts_vision_v2, deep_compound_prompts_text_v2 = self.prompt_learner()
        image_feats, text_feats = self.base_model(images, caption_ids, ctx, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, vision, shared_vision, deep_compound_prompts_vision_v2, deep_compound_prompts_text_v2, self.tokenized_prompts)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
            
        if 'triplet' in self.current_task:
            ret.update({'triplet_loss':objectives.compute_triplet(i_feats, t_feats)})
        
        if 'id' in self.current_task:

            image_logits = self.classifier(i_feats.float()).float()
            text_logits = self.classifier(t_feats.float()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
                  
        if 'mlm' in self.current_task:
            mlm_ids, mlm_labels = self._build_random_masked_tokens_and_labels_with_ctx(caption_ids, ctx)
                        
            deep_compound_prompts_text_cat = []
            for i in range(len(deep_compound_prompts_text)):
                cat_temp = torch.cat((deep_compound_prompts_text[i], deep_compound_prompts_text_v2[i]), dim=0)
                deep_compound_prompts_text_cat.append(cat_temp)
        
            new_ctx = torch.cat((ctx, shared_vision), dim=0)
            
            
            mlm_feats = self.base_model.encode_text(mlm_ids, new_ctx, deep_compound_prompts_text_cat)
            x = self.cross_former(mlm_feats.float(), image_feats.float(), image_feats.float())

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = mlm_labels.reshape(-1).to(scores.device)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    
    return model
