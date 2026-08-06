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
        # DDP normally receives already-embedded learnable prompts.  The
        # training-only prompt-free auxiliary branch also needs the original
        # CLIP token embedding in order to build fixed natural-language
        # prototypes.  Keep a non-persistent frozen copy so old DDP
        # checkpoints remain strictly loadable without adding checkpoint keys.
        self.register_buffer(
            "token_embedding_weight",
            clip_model.token_embedding.weight.detach().clone(),
            persistent=False,
        )


    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        row_indices = torch.arange(x.shape[0], device=x.device)
        eot_indices = tokenized_prompts.argmax(dim=-1).to(x.device)
        x = x[row_indices, eot_indices] @ self.text_projection
        return x

    def encode_tokenized(self, tokenized_prompts):
        """Encode fixed text with the same frozen CLIP text tower as DDP."""
        tokenized_prompts = tokenized_prompts.to(
            self.token_embedding_weight.device
        )
        prompts = F.embedding(
            tokenized_prompts, self.token_embedding_weight
        ).type(self.dtype)
        return self.forward(prompts, tokenized_prompts)


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
        self.feature_adapter = None
        self.feature_adapter_correction = "linear_residual"
        self.feature_adapter_bank = None
        self.transformer_adapter_bank = None

    def encode_prompt_free_image(
        self,
        image,
        normalize=True,
        return_tokens=False,
    ):
        """Training-only global CLIP route before visual-prompt insertion.

        This calls the *same* frozen image encoder owned by DDP with
        ``visual_prompts=None``.  For ViT-B/16, the returned global feature is
        therefore the ordinary projected CLIP CLS token.  The method is used
        only to train the shared Adapter; normal DDP inference does not call it.
        """
        token_features = self.image_encoder(image.type(self.dtype), None)
        if token_features.ndim != 3:
            raise RuntimeError(
                "The prompt-free auxiliary route requires a ViT image encoder "
                "returning [batch, tokens, feature_dim]"
            )
        if return_tokens:
            return token_features
        global_cls = token_features[:, 0, :].float()
        if normalize:
            global_cls = F.normalize(global_cls, dim=-1)
        return global_cls

    def encode_prompt_free_text(self, tokenized_prompts, normalize=True):
        """Encode fixed prototype text without DDP's learnable text prompts."""
        text_features = self.text_encoder.encode_tokenized(tokenized_prompts)
        text_features = text_features.float()
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def extract_three_route_features(
        self,
        image,
        cls_id=None,
        inference=False,
    ):
        """Expose the three representations used by the integrated design.

        ``global_cls`` is the training-only prompt-free auxiliary route.
        ``ddp_pooled`` is the original DDP token-attention route.
        ``prompted_cls`` is the internal Adapter transfer route.  This helper
        is intended for diagnostics; inference should call :meth:`forward`,
        which omits the prompt-free route and its extra encoder pass.
        """
        global_cls = self.encode_prompt_free_image(image)
        pooled, base_logits, text_features, prompted_cls = (
            self.extract_path_features(
                image,
                cls_id=cls_id,
                inference=inference,
                return_cls_features=True,
            )
        )
        return {
            "global_cls": global_cls,
            "ddp_pooled": pooled,
            "prompted_cls": prompted_cls,
            "base_path_logits": base_logits,
            "text_features": text_features,
        }

    def enable_feature_adapter(
        self,
        bottleneck_dim=128,
        residual_scale=0.1,
        correction_mode="linear_residual",
    ):
        from ddp_internal_adapter import (
            CORRECTION_MODES,
            SharedResidualFeatureAdapter,
        )

        if correction_mode not in CORRECTION_MODES:
            raise ValueError(
                f"Unknown correction mode '{correction_mode}'; expected one "
                f"of {CORRECTION_MODES}"
            )

        feature_dim = int(self.text_encoder.text_projection.shape[1])
        self.feature_adapter = SharedResidualFeatureAdapter(
            feature_dim=feature_dim,
            bottleneck_dim=bottleneck_dim,
            residual_scale=residual_scale,
        ).to(self.visual_prompts.device)
        self.feature_adapter_correction = correction_mode
        return self.feature_adapter

    def enable_task_adapter_bank(self, adapter_bank):
        """Attach a frozen class-routed Bank to the prompted CLS route."""
        if getattr(self, "transformer_adapter_bank", None) is not None:
            raise RuntimeError(
                "Disable the Transformer Adapter Bank before enabling the "
                "CLS Feature-Difference Bank"
            )
        if self.feature_adapter is not None:
            raise RuntimeError(
                "Disable the single feature_adapter before enabling an Adapter Bank"
            )
        if getattr(adapter_bank, "routing_mode", None) != "class_introduction_task":
            raise ValueError(
                "DDP only accepts deterministic class-introduction task routing"
            )
        if getattr(adapter_bank, "correction_mode", None) != "feature_difference":
            raise ValueError("DDP Task Adapter Bank requires Feature Difference")
        self.feature_adapter_bank = adapter_bank
        self.feature_adapter_bank.eval()
        for parameter in self.feature_adapter_bank.parameters():
            parameter.requires_grad_(False)
        return self.feature_adapter_bank

    def disable_task_adapter_bank(self):
        self.feature_adapter_bank = None

    def enable_transformer_adapter_bank(self, adapter_bank, freeze=True):
        """Attach a class-routed Adapter Bank inside visual ViT blocks."""
        if self.feature_adapter is not None or self.feature_adapter_bank is not None:
            raise RuntimeError(
                "Transformer and final-feature Adapters cannot be active together"
            )
        if getattr(adapter_bank, "routing_mode", None) != "class_introduction_task":
            raise ValueError(
                "DDP only accepts deterministic class-introduction task routing"
            )
        if getattr(adapter_bank, "adapter_location", None) != "parallel_to_vit_mlp":
            raise ValueError("Unsupported Transformer Adapter location")
        if int(getattr(adapter_bank, "hidden_dim", -1)) != self.width:
            raise ValueError(
                f"Transformer Adapter width must be {self.width}, got "
                f"{getattr(adapter_bank, 'hidden_dim', None)}"
            )
        self.transformer_adapter_bank = adapter_bank
        if freeze:
            self.transformer_adapter_bank.eval()
            for parameter in self.transformer_adapter_bank.parameters():
                parameter.requires_grad_(False)
        return self.transformer_adapter_bank

    def disable_transformer_adapter_bank(self):
        self.transformer_adapter_bank = None

    def _text_features(self, cls_id, inference):
        if inference:
            neg_feats = []
            pos_feats = []
            feature_device = self.visual_prompts.device
            for entry in self.text_feature_cache.values():
                neg_feats.append(entry['neg'].to(feature_device))
                pos_feats.append(entry['pos'].to(feature_device))
            return torch.cat(
                [torch.cat(neg_feats, dim=0), torch.cat(pos_feats, dim=0)],
                dim=0,
            )

        prompts, tokenized_prompts = self.prompt_learner(cls_id)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        K = cls_id[1] - cls_id[0]
        self.text_feature_cache[tuple(cls_id)] = {
            'neg': text_features[:K].detach().cpu(),
            'pos': text_features[K:].detach().cpu(),
        }
        return text_features

    def extract_path_features(
        self,
        image,
        cls_id=None,
        inference=False,
        return_cls_features=False,
    ):
        """Return DDP pooled path features and exact pre-adapter path logits."""
        text_features = self._text_features(cls_id, inference)
        cls_id_range = range(cls_id[0], cls_id[1])
        B = image.shape[0]
        K = len(cls_id_range)
        D = 768

        all_cls_ids = list(cls_id_range) + [i + self.n_cls for i in cls_id_range]
        visual_prompts_all = self.visual_prompts[all_cls_ids]
        visual_prompts_all = visual_prompts_all.unsqueeze(0).expand(B, -1, -1, -1)
        visual_prompts_all = visual_prompts_all.reshape(
            2 * K * B, self.l_vp, D
        )
        image_expand = image.unsqueeze(1).expand(
            -1, 2 * K, -1, -1, -1
        ).reshape(2 * K * B, *image.shape[1:])

        transformer_adapter_bank = getattr(
            self, "transformer_adapter_bank", None
        )
        adapter_path_task_ids = None
        if transformer_adapter_bank is not None:
            from emotic_transformer_adapter_bank import (
                path_task_ids_for_classes,
            )

            adapter_path_task_ids = path_task_ids_for_classes(
                list(cls_id_range),
                B,
                image_expand.device,
            )
        token_features = self.image_encoder(
            image_expand.type(self.dtype),
            visual_prompts_all,
            transformer_adapter_bank=transformer_adapter_bank,
            adapter_path_task_ids=adapter_path_task_ids,
        )
        token_features = token_features.permute(0, 2, 1)
        token_features = token_features / token_features.norm(
            dim=1, keepdim=True
        )
        feature_dim = token_features.shape[1]
        token_count = token_features.shape[2]
        token_features = token_features.view(
            B, 2 * K, feature_dim, token_count
        )

        token_logits = 20 * torch.einsum(
            'bkdn,kd->bkn', token_features, text_features
        )
        negative_weights = F.softmax(token_logits[:, K:, :], dim=-1)
        path_weights = torch.cat(
            [negative_weights, negative_weights], dim=1
        )
        pooled_features = torch.einsum(
            'bkdn,bkn->bkd', token_features, path_weights
        )
        base_path_logits = 5 * (token_logits * path_weights).sum(-1)
        if return_cls_features:
            cls_features = token_features[:, :, :, 0]
            return (
                pooled_features,
                base_path_logits,
                text_features,
                cls_features,
            )
        return pooled_features, base_path_logits, text_features

    def logits_from_path_features(
        self,
        pooled_features,
        base_path_logits,
        text_features,
        return_adapter_aux=False,
        ddp_pooled_features=None,
    ):
        path_logits = base_path_logits
        adapter_aux = None
        feature_adapter = getattr(self, "feature_adapter", None)
        feature_adapter_bank = getattr(self, "feature_adapter_bank", None)
        if feature_adapter is not None and feature_adapter_bank is not None:
            raise RuntimeError("A single Adapter and Adapter Bank cannot be active together")
        if feature_adapter_bank is not None:
            seen_classes = path_logits.shape[1] // 2
            correction = feature_adapter_bank.feature_difference_correction(
                pooled_features,
                text_features,
                seen_classes=seen_classes,
                logit_scale=100.0,
            )
            path_logits = path_logits.float() + correction
            adapter_aux = {
                "correction": correction,
                "correction_mode": "feature_difference",
                "feature_source": "prompted_cls",
                "max_adapter_task": feature_adapter_bank.max_task,
                "routing_mode": feature_adapter_bank.routing_mode,
                "classification_loss": feature_adapter_bank.classification_loss,
                "loss_config_sha256": feature_adapter_bank.loss_config_sha256,
                "inference_alpha": feature_adapter_bank.inference_alpha,
            }
        elif feature_adapter is not None:
            from ddp_internal_adapter import feature_logit_correction

            adapted, original = feature_adapter(pooled_features)
            correction_mode = getattr(
                self, "feature_adapter_correction", "linear_residual"
            )
            correction = feature_logit_correction(
                adapted,
                original,
                text_features,
                mode=correction_mode,
                logit_scale=100.0,
                pooled_features=(
                    pooled_features
                    if ddp_pooled_features is None
                    else ddp_pooled_features
                ),
            )
            path_logits = path_logits.float() + correction
            adapter_aux = {
                "adapted": adapted,
                "original": original,
                "correction": correction,
                "correction_mode": correction_mode,
            }
            if correction_mode == "feature_correction":
                from ddp_internal_adapter import (
                    norm_preserving_feature_correction,
                )

                source_pooled = (
                    pooled_features
                    if ddp_pooled_features is None
                    else ddp_pooled_features
                )
                adapter_aux["ddp_pooled"] = source_pooled.float()
                adapter_aux["corrected_pooled"] = (
                    norm_preserving_feature_correction(
                        source_pooled,
                        adapted.float() - original.float(),
                    )
                )

        batch, path_count = path_logits.shape
        logits = path_logits.reshape(batch, 2, path_count // 2)
        if return_adapter_aux:
            return logits, adapter_aux
        return logits

    def forward(
        self,
        image,
        cls_id=None,
        inference=False,
        return_adapter_aux=False,
    ):
        feature_adapter = getattr(self, "feature_adapter", None)
        feature_adapter_bank = getattr(self, "feature_adapter_bank", None)
        use_cls_feature_correction = (
            feature_adapter is not None
            and getattr(
                self, "feature_adapter_correction", "linear_residual"
            )
            == "feature_correction"
        )
        use_cls_adapter_bank = feature_adapter_bank is not None
        extracted = self.extract_path_features(
            image,
            cls_id=cls_id,
            inference=inference,
            return_cls_features=(
                use_cls_feature_correction or use_cls_adapter_bank
            ),
        )
        if use_cls_feature_correction or use_cls_adapter_bank:
            pooled, base_logits, text_features, cls_features = extracted
            adapter_features = cls_features
        else:
            pooled, base_logits, text_features = extracted
            adapter_features = pooled
        return self.logits_from_path_features(
            adapter_features,
            base_logits,
            text_features,
            return_adapter_aux=return_adapter_aux,
            ddp_pooled_features=(pooled if use_cls_feature_correction else None),
        )

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
