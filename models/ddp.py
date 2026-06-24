import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F
import time
import os
_tokenizer = _Tokenizer()

__all__ = ['ddp', 'DDP']


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    configured_path = os.path.expanduser(cfg.MLCCLIP.PRETRAINED_PATH)
    default_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "pretrained", "clip", os.path.basename(url))
    )

    if configured_path:
        if not os.path.isfile(configured_path):
            raise FileNotFoundError(f"CLIP model not found: {configured_path}")
        model_path = configured_path
    elif os.path.isfile(default_path):
        model_path = default_path
    else:
        model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class MLCPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_pos = cfg.TRAINER.COOP_MLC.N_CTX_POS
        n_ctx_neg = cfg.TRAINER.COOP_MLC.N_CTX_NEG
        ctx_init_pos = cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT.strip()
        ctx_init_neg = cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT.strip()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        if ctx_init_pos and ctx_init_neg:
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = clip.tokenize(ctx_init_pos)
            prompt_neg = clip.tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if cfg.TRAINER.COOP_MLC.CSC:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if cfg.TRAINER.COOP_MLC.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(n_cls, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_cls, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        print(f'Initial positive context: "{prompt_prefix_pos}"') # "X X X X X X X X X X X X X X X X"
        print(f'Initial negative  context: "{prompt_prefix_neg}"') # "X X X X X X X X X X X X X X X X"
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]
        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        for p_pos, p_neg in zip(prompts_pos, prompts_neg):
            tokenized_prompts_pos.append(clip.tokenize(p_pos))
            tokenized_prompts_neg.append(clip.tokenize(p_neg))

        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)

        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx_neg:, :])

        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def forward(self, cls_id=None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        low_range, high_range=cls_id

        if ctx_pos.dim() == 2:
            if cls_id is None:
                ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_pos = ctx_pos.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_pos = ctx_pos[low_range:high_range]

        if ctx_neg.dim() == 2:
            if cls_id is None:
                ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_neg = ctx_neg.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_neg = ctx_neg[low_range:high_range]
                

        if cls_id is None:
            prefix_pos = self.token_prefix_pos
            prefix_neg = self.token_prefix_neg
            suffix_pos = self.token_suffix_pos
            suffix_neg = self.token_suffix_neg
        else:
            prefix_pos = self.token_prefix_pos[low_range:high_range]
            prefix_neg = self.token_prefix_neg[low_range:high_range]
            suffix_pos = self.token_suffix_pos[low_range:high_range]
            suffix_neg = self.token_suffix_neg[low_range:high_range]

        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        prompts_neg = torch.cat(
            [
                prefix_neg,
                ctx_neg,
                suffix_neg,
            ],
            dim=1,
        )

        prompts = torch.cat([prompts_neg, prompts_pos], dim=0)
        if cls_id is not None:
            tokenized_prompts_pos = self.tokenized_prompts[self.n_cls:][low_range:high_range]
            tokenized_prompts_neg = self.tokenized_prompts[:self.n_cls][low_range:high_range]
            tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)
        else:
            tokenized_prompts = self.tokenized_prompts
        return prompts, tokenized_prompts


class DDP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.visual_encoder_type = cfg.MODEL.BACKBONE.NAME
        self.prompt_learner = MLCPromptLearner(cfg, classnames, clip_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = cfg.TRAINER.COOP_MLC.LS
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_cls = len(classnames)
        self.l_vp = 16
        self.width = 768

        self.visual_prompts = nn.Parameter(nn.init.normal_(torch.empty(self.n_cls*2, self.l_vp, self.width, dtype=self.dtype), std=0.02) )
        self.text_feature_cache = {}

    def forward(self, image, cls_id=None, inference=False):
        if inference:
            neg_feats = []
            pos_feats = []
            for entry in self.text_feature_cache.values():
                neg_feats.append(entry['neg'].to(self.device))
                pos_feats.append(entry['pos'].to(self.device))
            text_features = torch.cat([torch.cat(neg_feats, dim=0), torch.cat(pos_feats, dim=0)], dim=0)
        else:
            prompts, tokenized_prompts = self.prompt_learner(cls_id)
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            K = cls_id[1] - cls_id[0]
            self.text_feature_cache[tuple(cls_id)] = {
                'neg': text_features[:K].detach().cpu(),
                'pos': text_features[K:].detach().cpu(),
            }

        cls_id_range = range(cls_id[0],cls_id[1])
        B = image.shape[0]
        K = len(cls_id_range)
        D = 768
        l_vp = self.l_vp

        # visual prompts
        all_cls_ids = list(cls_id_range) + [i + self.n_cls for i in cls_id_range]
        visual_prompts_all = self.visual_prompts[all_cls_ids]

        visual_prompts_all = visual_prompts_all.unsqueeze(0).expand(B, -1, -1, -1)
        visual_prompts_all = visual_prompts_all.reshape(2 * K * B, l_vp, D)
        image_expand = image.unsqueeze(1).expand(-1, 2 * K, -1, -1, -1).reshape(2 * K * B, *image.shape[1:])

        # encode
        image_features_all = self.image_encoder(image_expand.type(self.dtype), visual_prompts_all)
        image_features_all = image_features_all.permute(0, 2, 1)
        image_features_all = image_features_all / image_features_all.norm(dim=1, keepdim=True)

        D_actual = image_features_all.shape[1]
        N_actual = image_features_all.shape[2]
        image_features_all = image_features_all.view(B, 2 * K, D_actual, N_actual)

        output_all = 20 * torch.einsum('bkdn,kd->bkn', image_features_all, text_features)

        output_pos = output_all[:, :K, :]
        output_neg = output_all[:, K:, :]

        output_list = torch.cat([output_pos, output_neg], dim=1)

        b, c, _ = output_list.shape
        output_half = output_list[:,  c // 2:]
        w_half = F.softmax(output_half, dim=-1) 
        w = torch.cat([w_half, w_half], dim=1)
        output_list = 5 * (output_list * w).sum(-1)
        b, c = output_list.shape
        logits = output_list.resize(b, 2, c//2)

        return logits

    @property
    def network_name(self):
        name = ''
        name += 'DDP-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params


def ddp(cfg, classnames, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building DDP")
    model = DDP(cfg, classnames, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    return model
