from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from torch import nn


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        ctx, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, vision, shared_vision, deep_compound_prompts_vision_v2, deep_compound_prompts_text_v2 = model.prompt_learner()
        
        deep_compound_prompts_text_cat = []
        for i in range(len(deep_compound_prompts_text)):
            cat_temp = torch.cat((deep_compound_prompts_text[i], deep_compound_prompts_text_v2[i]), dim=0)
            deep_compound_prompts_text_cat.append(cat_temp)
        new_ctx = torch.cat((ctx, shared_vision), dim=0)
        
        for pid, caption in self.txt_loader:
            ## ===> Generate new token_ids after adding prompt
            prompt_len = ctx.shape[0] * 2
            max_len = caption.shape[1]
            new_id = model.tokenized_prompts.flatten()[1:prompt_len+1]
            for idx, text_id in enumerate(caption):
                if text_id.argmax(dim=-1) < (len(caption[idx]) - prompt_len):
                    # text_id 可以直接拼接 prompt_id
                    caption[idx] = torch.cat((text_id[0].unsqueeze(0), new_id.to(text_id), text_id[1:-prompt_len]), dim=-1)
                else:
                    # 需要先截断，去掉末尾的两个word，再拼接 [BOS] + prompt_id + tunc_raw_id + [EOS]
                    max_idx = text_id.argmax(dim=-1)
                    tunc_idx = max_len - prompt_len - 1
                    eos_id = text_id[max_idx]

                    if eos_id.dim() == 0:
                        eos_id = eos_id.unsqueeze(0)

                    suffix_id = text_id[1:tunc_idx]
                    if suffix_id.dim() == 0:
                        suffix_id = suffix_id.unsqueeze(0)     

                    caption[idx] = torch.cat((text_id[0].unsqueeze(0), new_id.to(text_id), 
                                            suffix_id, eos_id), dim=-1)
            ## <=== End of Generating new token_ids after adding prompt
            
            caption = caption.to(device)
            with torch.no_grad():
                # text_feat = model.encode_text(caption)       
                text_feat = model.base_model.encode_text(caption, new_ctx, deep_compound_prompts_text_cat)
                text_feat = text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        

        # image
        vision_first_layer_cat = torch.cat((shared_ctx, vision), dim=0)
        deep_compound_prompts_vision_cat = []
        for i in range(len(deep_compound_prompts_vision)):
            deep_compound_prompts_vision_cat.append(torch.cat((deep_compound_prompts_vision[i], deep_compound_prompts_vision_v2[i]), dim=0))
            
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                # img_feat = model.encode_image(img)
                img_feat = model.base_model.encode_image(img, vision_first_layer_cat, deep_compound_prompts_vision_cat)
                img_feat = img_feat[:, 0, :].float()
                
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]
