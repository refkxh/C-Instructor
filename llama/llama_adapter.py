import os
import json
from pathlib import Path

import clip
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .llama import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download


# class HistoryEmbeddings(nn.Module):
#     def __init__(self, image_feat_size=768, angle_feat_size=4, hidden_size=768, max_action_steps=30):
#         super().__init__()
#         self.ang_linear = nn.Linear(angle_feat_size, hidden_size)
#         self.ang_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

#         self.num_h_pano_layers = 2
#         self.pano_num_heads = 12
#         self.pano_mlp_ratio = 4.0

#         if self.num_h_pano_layers > 0:
#             self.pano_img_linear = nn.Linear(image_feat_size, hidden_size)
#             self.pano_img_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
#             self.pano_ang_linear = nn.Linear(angle_feat_size, hidden_size)
#             self.pano_ang_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
#             self.pano_encoder = nn.Sequential()
#             for i in range(self.num_h_pano_layers):
#                 self.pano_encoder.add_module(f'block_{i}', Block(hidden_size, self.pano_num_heads, self.pano_mlp_ratio, qkv_bias=True))
#         else:
#             self.pano_encoder = None

#         self.position_embeddings = nn.Embedding(max_action_steps, hidden_size)

#         # tf naming convention for layer norm
#         self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, projected_img_feats, ang_feats, pano_img_feats, pano_ang_feats, pos_ids=None):

#         if projected_img_feats is not None:
#             embeddings = projected_img_feats + self.ang_layer_norm(self.ang_linear(ang_feats))

#             if self.pano_encoder is not None:
#                 batch_size, num_steps, num_pano, _ = pano_img_feats.size()
#                 pano_img_feats = pano_img_feats.view(batch_size*num_steps, num_pano, -1)
#                 pano_ang_feats = pano_ang_feats.view(batch_size*num_steps, num_pano, -1)
#                 pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) + \
#                                   self.pano_ang_layer_norm(self.pano_ang_linear(pano_ang_feats))
#                 pano_embeddings = self.pano_encoder(pano_embeddings)
                
#                 pano_embeddings = pano_embeddings.view(batch_size, num_steps, num_pano, -1)
#                 pano_embeddings = torch.mean(pano_embeddings, 2)
                
#                 embeddings = embeddings + pano_embeddings

#             if pos_ids is not None:
#                 embeddings = embeddings + self.position_embeddings(pos_ids)
#                 embeddings = self.layer_norm(embeddings)
#                 embeddings = self.dropout(embeddings)
#         else:
#             embeddings = None

#         return embeddings


class HistoryEmbeddingsV2(nn.Module):
    def __init__(self, angle_feat_size=4, hidden_size=768, max_action_steps=30):
        super().__init__()
        self.ang_linear = nn.Linear(angle_feat_size, hidden_size)
        self.ang_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.position_embeddings = nn.Embedding(max_action_steps, hidden_size)
        self.cls_embeddings = nn.Embedding(2, hidden_size)

    def forward(self, projected_img_feats, ang_feats, pano_ang_feats, pos_ids=None):

        if projected_img_feats is not None:
            ang_feats = ang_feats.unsqueeze(2)  # B, num_steps, 1, C
            ang_feats = torch.cat([ang_feats, pano_ang_feats], dim=2)  # B, num_steps, 1+num_pano, C

            batch_size, num_steps, num_pano, _ = ang_feats.size()
            ang_feats = ang_feats.view(batch_size*num_steps, num_pano, -1)
            ang_feats = self.ang_layer_norm(self.ang_linear(ang_feats.float()))
            ang_feats = ang_feats.view(batch_size, num_steps, num_pano, -1)
            
            embeddings = projected_img_feats + ang_feats

            if pos_ids is not None:
                embeddings = embeddings + self.position_embeddings(pos_ids).unsqueeze(2)

            cls_ids = torch.zeros(num_pano, dtype=int).expand((1, -1)).to(projected_img_feats.device)
            cls_ids[0, 0] = 1
            embeddings = embeddings + self.cls_embeddings(cls_ids).unsqueeze(1)
        else:
            embeddings = None

        return embeddings


class LLaMA_adapter(nn.Module):

    def __init__(self, llama_ckpt_dir, llama_tokenizer,
                 max_seq_len=512, max_batch_size=1,
                 clip_model='ViT-L/14',
                 v_embed_dim=768, v_depth=8,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=10, query_layer=31,
                 w_bias=False, 
                 w_lora=False, lora_rank=16, 
                 w_new_gate=False,
                 phase="finetune"):
        super().__init__()

        # load llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        w_bias = phase == "finetune"
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        ) # max_batch_size only affects inferenc

        # 1. clip and clip projector
        self.clip, self.clip_transform = clip.load(clip_model)

        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        # observation projector
        self.clip_ob_proj = nn.Linear(clip_dim, model_args.dim)
        self.clip_ob_proj_norm = nn.LayerNorm(model_args.dim)
        self.ob_ang_linear = nn.Linear(4, model_args.dim)
        self.ob_ang_layer_norm = nn.LayerNorm(model_args.dim, eps=1e-12)

        # 1.5. history embeddings
        self.max_action_steps = 30

        # self.history_embeddings = HistoryEmbeddings(image_feat_size=clip_dim, hidden_size=v_embed_dim)
        self.history_embeddings = HistoryEmbeddingsV2(hidden_size=v_embed_dim, max_action_steps=self.max_action_steps)

        self.query_len = query_len
        self.query_layer = query_layer

        # 2. visual query, blocks and projector
        self.visual_query = nn.Embedding(query_len, v_embed_dim)
        self.visual_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)])
        self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
        self.visual_proj_norm = nn.LayerNorm(model_args.dim)

        # observation query
        self.ob_query = nn.Embedding((36 + 1), model_args.dim)
        # action query
        self.action_query = nn.Embedding(1, model_args.dim)
        # logits temperature
        self.logits_temp = torch.nn.Parameter(torch.zeros(1))

        # 3. adapter query
        self.adapter_query = nn.Embedding(query_len * query_layer, model_args.dim)
        # observation query
        # self.ob_query = nn.Embedding(36 * query_layer, model_args.dim)

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 5. llama
        model_args.w_bias = w_bias
        model_args.w_lora = w_lora
        model_args.lora_rank = lora_rank
        model_args.w_new_gate = w_new_gate
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)

        del self.clip.transformer

         # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.criterion_no_ignore = torch.nn.CrossEntropyLoss()

        # 7. training parameters
        self.phase = phase
        self.get_trainable_params(self.phase)

        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            train_param_name = ['gate', 'clip_proj', 'clip_proj_norm', 'clip_ob_proj', 'clip_ob_proj_norm', 'ob_ang_linear', 'ob_ang_layer_norm',
                                'visual_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query', 'ob_query', 'action_query',
                                'history_embeddings', 'logits_temp']
            for name, para in self.named_parameters():
                if name.startswith("llama.layers"):
                    layer_num = int(name.split('.')[2])
                    if layer_num >= 30:
                        para.data = para.data.float()
                        para.requires_grad = True

                if name.startswith("llama.norm"):
                    para.data = para.data.float()
                    para.requires_grad = True

                # if name.startswith("llama."):
                #     if 'norm' in name or 'bias' in name:
                #         para.data = para.data.float()
                #         para.requires_grad = True

                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        elif phase == 'pretrain':
            train_param_name = ['gate', 'clip_proj', 'clip_proj_norm', 'visual_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query']
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True
        
        else:
            raise ValueError(f"Unknown model phase: {phase}")
        
    def clip_encode_image(self, x):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x

    def forward_visual(self, imgs, ang_feats=None, pano_img_feats=None, pano_ang_feats=None, raw_input=False):
        if not raw_input:
            clip_feats = imgs.unsqueeze(2)  # B, num_steps, 1, C
            clip_feats = torch.cat([clip_feats, pano_img_feats], dim=2)  # B, num_steps, 1+num_pano, C

            batch_size, num_steps, num_pano, _ = clip_feats.size()
            clip_feats = clip_feats.view(batch_size*num_steps, num_pano, -1)
            clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))
            clip_feats = clip_feats.view(batch_size, num_steps, num_pano, -1)

            # clip_feats = imgs
            # clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

            hist_step_ids = torch.arange(num_steps).expand((1, -1)).to(clip_feats.device)

            # clip_feats = self.history_embeddings(clip_feats, ang_feats, pano_img_feats, pano_ang_feats, hist_step_ids)
            clip_feats = self.history_embeddings(clip_feats, ang_feats, pano_ang_feats, hist_step_ids)

            clip_feats = clip_feats.view(batch_size*num_steps, num_pano, -1)
            visual_query = self.visual_query.weight.unsqueeze(0).repeat(batch_size*num_steps, 1, 1)

            visual_query = torch.cat([visual_query, clip_feats], dim=1)
            for block in self.visual_blocks:
                visual_query = block(visual_query)

            visual_query = visual_query[:, :self.query_len, :]
            visual_query = self.visual_proj(visual_query)
            visual_query = self.visual_proj_norm(visual_query)
            visual_query = visual_query.view(batch_size, num_steps, self.query_len, -1)
        else:
            clip_feats = self.clip_encode_image(imgs)
            clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

            visual_query = self.visual_query.weight.unsqueeze(
                0).repeat(len(imgs), 1, 1)
                
            visual_query = torch.cat([visual_query, clip_feats], dim=1)
            for block in self.visual_blocks:
                visual_query = block(visual_query)

            visual_query = visual_query[:, :self.query_len, :]
            visual_query = self.visual_proj(visual_query)
            visual_query = self.visual_proj_norm(visual_query)

        return visual_query

    def forward(self, tokens, labels, imgs=None, ang_feats=None, pano_img_feats=None, pano_ang_feats=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, ob_id_seps=None, ob_action_viewindex=None):
        if ob_img_feats is None:
            raw_input = ang_feats is None
            visual_query = self.forward_visual(imgs, ang_feats, pano_img_feats, pano_ang_feats, raw_input)
        else:
            ob_feats = self.clip_ob_proj_norm(self.clip_ob_proj(ob_img_feats.float()))
            ob_feats = ob_feats + self.ob_ang_layer_norm(self.ob_ang_linear(ob_ang_feats))
            ob_feats = ob_feats + self.ob_query.weight
            if imgs is None:
                visual_query = None
            else:
                visual_query = self.forward_visual(imgs, ang_feats, pano_img_feats, pano_ang_feats, raw_input=False)

        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)

        if ob_img_feats is not None:
            for i in range(_bsz):
                h[i, ob_id_seps[i][:-1]] = ob_feats[i, ob_nav_types[i]>0].type_as(h)
                h[i, ob_id_seps[i][-1]] = self.action_query.weight.squeeze(0).type_as(h)

        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        # ob_query = self.ob_query.weight.reshape(self.query_layer, 36, -1).unsqueeze(1)
        adapter_index = 0
        for i, layer in enumerate(self.llama.layers[-1 * self.query_layer:]):
            if ob_img_feats is not None and i == self.query_layer - 2:
                for j in range(_bsz):
                    h[j, ob_id_seps[j][:-1]] = ob_feats[j, ob_nav_types[j]>0].type_as(h)
                    h[j, ob_id_seps[j][-1]] = self.action_query.weight.squeeze(0).type_as(h)

            if ang_feats is not None:
                batch_size, num_steps, _, _ = visual_query.size()
                visual_query_reshape = visual_query.view(batch_size*num_steps, self.query_len, -1)
                dynamic_adapter = adapter[adapter_index].repeat(batch_size*num_steps, 1, 1)
                dynamic_adapter = dynamic_adapter + visual_query_reshape
                dynamic_adapter = dynamic_adapter.view(batch_size, num_steps*self.query_len, -1)
            elif visual_query is not None:
                dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
                dynamic_adapter = dynamic_adapter + visual_query
            else:
                dynamic_adapter = None

            # if ob_img_feats is None:
            #     ob_adapter = None
            # else:
            #     ob_adapter = ob_query[adapter_index].repeat(_bsz, 1, 1)
            #     ob_adapter = ob_adapter + ob_feats

            h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)

        if ob_img_feats is None:
            output = self.llama.output(h)
            output = output[:, :-1, :]
            labels = labels[:, 1:]

            if labels.sum() == 0:
                c_loss = output.mean() * 0
            else:
                assert self.llama.vocab_size == 32000
                c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())
        else:
            output_features = []
            for i in range(_bsz):
                output_features.append(h[i, ob_id_seps[i][-1]])
            output_features = torch.stack(output_features, dim=0)
            output_features = output_features.unsqueeze(1)
            normalized_output_features = output_features / torch.linalg.vector_norm(output_features, dim=-1, keepdim=True)

            detached_ob_feats = ob_feats.detach()
            normalized_ob_feats = detached_ob_feats / torch.linalg.vector_norm(detached_ob_feats, dim=-1, keepdim=True)

            similarity = torch.matmul(normalized_output_features, normalized_ob_feats.transpose(1, 2)).squeeze(1)
            for i in range(_bsz):
                similarity[i, ob_nav_types[i]==0] = -1
            similarity = similarity * torch.exp(self.logits_temp)

            c_loss = self.criterion_no_ignore(similarity, ob_action_viewindex)

        return c_loss, c_loss

    @torch.inference_mode()
    def forward_inference(self, visual_query, tokens, start_pos: int, ob_feats=None, ob_nav_types=None, ob_id_seps=None):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)

        if ob_feats is not None:
            for i in range(_bsz):
                h[i, ob_id_seps[i][:-1]] = ob_feats[i, ob_nav_types[i]>0].type_as(h)
                h[i, ob_id_seps[i][-1]] = self.action_query.weight.type_as(h)

        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        # ob_query = self.ob_query.weight.reshape(self.query_layer, 36, -1).unsqueeze(1)
        adapter_index = 0
        for i, layer in enumerate(self.llama.layers[-1 * self.query_layer:]):
            if ob_feats is not None and i == self.query_layer - 2:
                for j in range(_bsz):
                    h[j, ob_id_seps[j][:-1]] = ob_feats[j, ob_nav_types[j]>0].type_as(h)
                    h[j, ob_id_seps[j][-1]] = self.action_query.weight.squeeze(0).type_as(h)

            if visual_query is None:
                dynamic_adapter = None
            elif len(visual_query.shape) == 4:
                batch_size, num_steps, _, _ = visual_query.size()
                visual_query_reshape = visual_query.view(batch_size*num_steps, self.query_len, -1)
                dynamic_adapter = adapter[adapter_index].repeat(batch_size*num_steps, 1, 1)
                dynamic_adapter = dynamic_adapter + visual_query_reshape
                dynamic_adapter = dynamic_adapter.view(batch_size, num_steps*self.query_len, -1)
            else:
                dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
                dynamic_adapter = dynamic_adapter + visual_query

            # if ob_feats is None:
            #     ob_adapter = None
            # else:
            #     ob_adapter = ob_query[adapter_index].repeat(_bsz, 1, 1)
            #     ob_adapter = ob_adapter + ob_feats

            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)

        if ob_feats is None:
            output = self.llama.output(h[:, -1, :])
            return output.float()
        else:
            output_features = []
            for i in range(_bsz):
                output_features.append(h[i, ob_id_seps[i][-1]])
            output_features = torch.stack(output_features, dim=0)
            output_features = output_features.unsqueeze(1)
            normalized_output_features = output_features / torch.linalg.vector_norm(output_features, dim=-1, keepdim=True)

            detached_ob_feats = ob_feats.detach()
            normalized_ob_feats = detached_ob_feats / torch.linalg.vector_norm(detached_ob_feats, dim=-1, keepdim=True)

            similarity = torch.matmul(normalized_output_features, normalized_ob_feats.transpose(1, 2)).squeeze(1)
            for i in range(_bsz):
                similarity[i, ob_nav_types[i]==0] = -1

            return similarity.float()


    @torch.inference_mode()
    def generate(
        self, imgs=None, prompts=None,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
        ang_feats=None, 
        pano_img_feats=None, 
        pano_ang_feats=None,
        ob_img_feats=None, 
        ob_ang_feats=None, 
        ob_nav_types=None, 
        ob_id_seps=None
    ):
        bsz = len(prompts)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        if imgs is not None:
            assert len(imgs) == len(prompts)

        ob_feats = None
        with torch.cuda.amp.autocast():
            if ob_img_feats is None:
                raw_input = ang_feats is None
                visual_query = self.forward_visual(imgs, ang_feats, pano_img_feats, pano_ang_feats, raw_input)
            else:
                ob_feats = self.clip_ob_proj_norm(self.clip_ob_proj(ob_img_feats.float()))
                ob_feats = ob_feats + self.ob_ang_layer_norm(self.ob_ang_linear(ob_ang_feats))
                ob_feats = ob_feats + self.ob_query.weight
                if imgs is None:
                    visual_query = None
                else:
                    visual_query = self.forward_visual(imgs, ang_feats, pano_img_feats, pano_ang_feats, raw_input=False)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
            if ob_feats is not None:
                tokens[k, len(t)] = 0
        
        if ob_feats is not None:
            actions = []
            with torch.cuda.amp.autocast():
                for i in range(bsz):
                    similarity = self.forward_inference(visual_query[i:i+1], tokens[i:i+1, :len(prompts[i])+1], 0, ob_feats[i:i+1], ob_nav_types[i:i+1], ob_id_seps[i:i+1])
                    action = torch.argmax(similarity, dim=-1)
                    actions.append(action.item())
            return actions
        else:
            input_text_mask = tokens != self.tokenizer.pad_id
            start_pos = min_prompt_size
            prev_pos = 0
            for cur_pos in range(start_pos, total_len):
                with torch.cuda.amp.autocast():
                    logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos, ob_feats, ob_nav_types, ob_id_seps)
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits, dim=-1)
                next_token = next_token.reshape(-1)

                next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
                tokens[:, cur_pos] = next_token
                # trick: early stop if bsz==1
                if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                    break
                prev_pos = cur_pos

            decoded = []
            for i, t in enumerate(tokens.tolist()):

                # cut to max gen len
                t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
                # cut to eos tok if any
                try:
                    t = t[: t.index(self.tokenizer.eos_id)]
                except ValueError:
                    pass
                decoded.append(self.tokenizer.decode(t))

            return decoded


_MODELS = {
    "BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    "LORA-BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth",
    "CAPTION-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/5088aeb63a89746b90bcfd5cb819e1c7411b2771b267c6d131ce73e250a8abf0_CAPTION-7B.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}

def available_models():
    return list(_MODELS.keys())

def load(name, llama_dir, device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts', max_seq_len=512,
        phase="finetune", max_batch_size=1):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}"), None

    # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
    llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})

    model = LLaMA_adapter(
        llama_ckpt_dir, llama_tokenzier_path,
        max_seq_len=max_seq_len, max_batch_size=max_batch_size,
        clip_model='ViT-L/14',
        v_embed_dim=768, v_depth=8,
        v_num_heads=16, v_mlp_ratio=4.0,
        query_len=10, query_layer=31,
        w_bias=model_cfg.get('w_bias', False), 
        w_lora=model_cfg.get('w_lora', False), 
        lora_rank=model_cfg.get('lora_rank', 16),
        w_new_gate=model_cfg.get('w_lora', False), # for compatibility
        phase=phase)

    load_result = model.load_state_dict(ckpt['model'], strict=False)

    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device), model.clip_transform
