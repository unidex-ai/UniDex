"""
PiZero model migrated from open-pi-zero, adapted for current project structure.
"""
from typing import Optional, Tuple

import hydra
import torch
from torch import nn
import numpy as np

from src.utils.kv_cache import KVCache
from src.unidex.modules import (
    ActionEncoder,
    SinusoidalPosEmb,
)
from src.utils.processing import VLAProcessor

from transformers import AutoTokenizer

class PointCloudUniDex(nn.Module):
    def __init__(
        self,
        cond_steps=1,
        horizon_steps=50,
        action_dim=48,
        proprio_dim=48,
        max_seq_len=532,
        max_pointcloud_text_tokens=532,
        final_action_clip_value=1.0,
        pcd_pos_embed=False,
        flow_sig_min=0.001,
        tokenizer_padding="max_length",
        use_lm_head=False,
        num_inference_steps=10,
        pcd_token_index=257152,
        vocab_size=257216,
        pad_token_id=0,
        time_hidden_size=256,
        time_max_period=100.0,
        action_expert_rope_theta=100.0,
        action_expert_adaptive_mode=None,
        pretrained_model_path="google/paligemma-3b-pt-224",
        quantize=False,
        lora=False,
        lora_r=32,
        lora_dropout=0.0,
        mixture=None,
        projector=None,
        joint=None,
        pointcloud_encoder=None,
        **kwargs
    ):
        super().__init__()
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.max_seq_len = max_seq_len
        self.pcd_pos_embed = pcd_pos_embed
        self.tokenizer_padding = tokenizer_padding
        self.max_pointcloud_text_tokens = max_pointcloud_text_tokens or max_seq_len
        self.pcd_token_index = pcd_token_index
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.time_hidden_size = time_hidden_size
        self.time_max_period = time_max_period
        self.action_expert_rope_theta = action_expert_rope_theta
        self.action_expert_adaptive_mode = action_expert_adaptive_mode
        self.pretrained_model_path = pretrained_model_path
        self.quantize = quantize
        self.lora = lora
        self.lora_r = lora_r
        self.lora_dropout = lora_dropout
        self.mixture = mixture
        self.projector = projector
        self.joint = joint
        self.pointcloud_encoder = pointcloud_encoder
        self.use_lm_head = use_lm_head

        self.max_poincloud_text_tokens = max_pointcloud_text_tokens
        self.num_proprio_tokens = cond_steps
        self.num_action_tokens = horizon_steps

        self.total_num_tokens = (
            self.max_poincloud_text_tokens
            + self.num_proprio_tokens
            + self.num_action_tokens
        )

        self.pcd_text_hidden_size = mixture.vlm.hidden_size
        self.proprio_hidden_size = mixture.proprio.hidden_size
        self.action_hidden_size = mixture.action.hidden_size

        # Action parameterization
        self.num_inference_steps = num_inference_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.final_action_clip_value = final_action_clip_value
        self.flow_sig_min = flow_sig_min

        # text input only
        self.embed_tokens = nn.Embedding(
            vocab_size,
            self.pcd_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

        # pointcloud encoder
        self.pointcloud_encoder = pointcloud_encoder
        self.multi_modal_projector = projector
        self.pointcloud_tokens = self.pointcloud_encoder.output_shape[0]

        # Mixtures
        self.joint_model = joint

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = action_expert_adaptive_mode
        if action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = SinusoidalPosEmb(
                time_hidden_size, time_max_period
            )
        else:  # matching pi0
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(
                self.action_hidden_size, time_max_period
            )
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )

        # Action decoder
        self.action_decoder = nn.Linear(
            self.action_hidden_size,
            self.action_dim,
        )

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.pcd_text_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights

            
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            tokenizer=self.tokenizer,
            num_pcd_tokens=self.pointcloud_tokens,
            max_seq_len=self.max_seq_len,
            tokenizer_padding=self.tokenizer_padding,
        )

        self.freeze_unused_weights()

    def load_pretrained(self, *args, **kwargs):
        """pointcloud_encoder, lm from paligemma"""
        import glob
        import os

        from safetensors import safe_open

        # load tensors from files
        safetensors_files = glob.glob(
            os.path.join(self.pretrained_model_path, "*.safetensors")
        )
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if "embed_tokens" in k:
                new_key = k.replace("language_model.model.embed_tokens.", "")
                embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)

        # load lm --- do not change any lora weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in (
            joint_model_state_dict.keys()
        ):  # avoid RuntimeError: OrderedDict mutated during iteration
            if "lora_" in key:
                lora_keys.append(key)
        for key in lora_keys:
            del joint_model_state_dict[key]
        for k, v in tensors.items():
            if "language_model.model" in k:
                new_key = k.replace("language_model.model.", "mixtures.vlm.")
                joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)

        # load pointcloud encoder
        pointcloud_encoder_state_dict = torch.load(self.pointcloud_encoder.pretrained_model_path, map_location="cpu", weights_only=False)
        self.pointcloud_encoder.load_state_dict(pointcloud_encoder_state_dict, strict=False)

    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        """no need to train vlm parameters after attention of last layer"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
            or f"{last_hidden_layer_index}.self_attn.v_proj" in name
        ):  # final norm is not initialized
            return True
        return False

    def freeze_unused_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.embed_tokens.weight.requires_grad = False
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = False

    def freeze_all_weights(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def tie_action_proprio_weights(self):
        """technically more than just tying weights"""
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    def build_text_cache(self):
        return KVCache()

    def get_optim_params(self) -> dict:
        return self.parameters()

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor]:
        """
        block attention --- padding for unused text tokens

                 pcd/text pcd/text pcd/text (padding) proprio action action
        pcd/text    x        x        x
        pcd/text    x        x        x
        pcd/text    x        x        x
        (padding)
        proprio     x        x        x                 x
        action      x        x        x                 x       x      x
        action      x        x        x                 x       x      x
        """
        bsz = attention_mask.size(0)
        proprio_start = self.max_poincloud_text_tokens
        proprio_end = self.max_poincloud_text_tokens + self.num_proprio_tokens
        action_start = proprio_end
        pcd_text_token_cnts = torch.sum(attention_mask, dim=1)
        causal_mask = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=attention_mask.device,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx, cnt in enumerate(pcd_text_token_cnts):
            causal_mask[idx, :cnt, :cnt] = 0  # pcd/text attend to itself
            causal_mask[idx, proprio_start:, :cnt] = (
                0  # proprio/action attend to pcd/text
            )
        causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
            0  # proprio attend to itself
        )
        causal_mask[:, action_start:, proprio_start:] = (
            0  # action attend to itself and proprio
        )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        if self.pcd_pos_embed:
            vlm_position_ids = torch.arange(
                1, 
                self.max_pointcloud_text_tokens + 1,
                dtype=torch.long,
                device=attention_mask.device,
            ).repeat(
                bsz, 1
            )
        else:
            pointcloud_tokens = self.pointcloud_tokens
            text_tokens = self.max_poincloud_text_tokens - pointcloud_tokens
            pointcloud_position_ids = torch.full(
                (pointcloud_tokens,), 
                2, 
                dtype=torch.long,
                device=attention_mask.device,
            )
            # Here all pointcloud tokens share the same position id, since the pointcloud encoder is permutation invariant using knn except for first token.
            pointcloud_position_ids[0] = 1

            text_position_ids = torch.arange(
                3, 
                text_tokens + 3, 
                dtype=torch.long,
                device=attention_mask.device,
            )
            vlm_position_ids = torch.cat([pointcloud_position_ids, text_position_ids], dim=0).repeat(bsz, 1)

        proprio_position_ids = torch.arange(
            1, 
            self.num_proprio_tokens + 1,
            device=attention_mask.device,
        ).repeat(
            bsz, 1
        )
        action_position_ids = torch.arange(
            self.num_proprio_tokens + 1,
            self.num_proprio_tokens + self.num_action_tokens + 1,
            device=attention_mask.device,
        ).repeat(bsz, 1)
        # since proprio and action share the same mixture weights, makes sense to use [1 (proprio), 2 (action), 3 (action), ...] instead of [1 (proprio), 1 (action), 2 (action), ...]
        return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        pcd_text_proprio_mask = causal_mask[
            ...,
            : self.max_poincloud_text_tokens + self.num_proprio_tokens,
            : self.max_poincloud_text_tokens + self.num_proprio_tokens,
        ]
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return pcd_text_proprio_mask, action_mask

    def build_causal_mask_and_position_ids_for_text(
        self,
        q_len: int,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        dtype, device = attention_mask.dtype, attention_mask.device
        bsz = attention_mask.size(0)

        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token, because we're in the prefill phase
            # assume no padding
            causal_mask = torch.full((bsz, q_len, q_len), 0, dtype=dtype, device=device)
        else:
            assert q_len == 1, "Using KV cache so should only use one single token"
            kv_len = kv_cache.num_items() + q_len
            # also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # this only works when we have no padding
            causal_mask = torch.full(
                (bsz, q_len, kv_len), 0, dtype=dtype, device=device
            )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # use the last location
            position_ids = attention_mask.cumsum(-1)[:, -1:]
        else:
            # create position_ids based on the size of the attention_mask
            # for padded tokens, use number 1
            position_ids = (attention_mask.cumsum(-1)).masked_fill_(
                (attention_mask == 0), 1
            )
        return causal_mask, position_ids

    # ---------- Inference ----------#

    def _forward_pointcloud_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pointcloud: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pointcloud.dtype, pointcloud.device

        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]\
        inputs_embeds = self.embed_tokens(input_ids)

        # pcd features from pointcloud encoder
        # [Batch_Size, Seq_Len, Number of Points, 6] => [Batch_Size, Seq_Len, Embed_Dim]
        B, S, N, _ = pointcloud.shape
        selected_pcd_feature = self.pointcloud_encoder(pointcloud.view(B * S, N, -1)) # [B * S, Number of Tokens, Embed_Dim]
        _, T, _ = selected_pcd_feature.shape
        selected_pcd_feature = selected_pcd_feature.view(B, S * T, -1)  # [B, Seq_Len, Embed_Dim]
        pcd_features = self.multi_modal_projector(selected_pcd_feature)

        # normalize the pcd features
        _, _, embed_dim = pcd_features.shape
        bsz, seq_len = input_ids.shape
        scaled_pcd_features = pcd_features / (self.pcd_text_hidden_size**0.5)

        # put embedding together - pcd, text, padding
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), self.pad_token_id, dtype=dtype, device=device
        )

        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.pcd_token_index) & (
            input_ids != self.pad_token_id
        )
        pcd_mask = input_ids == self.pcd_token_index
        final_embedding[text_mask] = inputs_embeds[text_mask]
        for i in range(bsz):
            pcd_indices = pcd_mask[i].nonzero(as_tuple=True)[0]
            num_pcd_tokens = len(pcd_indices)
            final_embedding[i, pcd_indices] = scaled_pcd_features[
                i, :num_pcd_tokens
            ]
        return final_embedding

    def infer_action(
        self,
        input_ids: torch.LongTensor,
        pointcloud: torch.FloatTensor,
        pcd_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pointcloud.dtype, pointcloud.device
        bsz = pointcloud.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the pcd tokens
        inputs_embeds = self._forward_pointcloud_and_text_embedding(input_ids, pointcloud)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # forward pass thru the vlm and proprio, cache the kv
        _, kv_caches = self.joint_model(
            attention_mask=pcd_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    def compile(self):
        self.joint_model.compile()
        self.pointcloud_encoder.compile()
        self.multi_modal_projector.compile()
        self.proprio_encoder = torch.compile(self.proprio_encoder)
        self.action_encoder = torch.compile(self.action_encoder)
        self.action_decoder = torch.compile(self.action_decoder)

    @torch.compile
    def guided_inference_iter(
        self,
        action: torch.FloatTensor,
        t: torch.FloatTensor,
        target_action: torch.FloatTensor,
        delta_t: float,
        beta: float,
        inpaint_attention: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        action_position_ids: torch.LongTensor,
        kv_caches: dict,
    ):
        time_cond = self.time_embedding(t)
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(action)
        else:
            action_embeds = self.action_encoder(action, time_cond)
        action_embeds = self.joint_model(
            attention_mask=action_mask,
            position_ids_all={"action": action_position_ids},
            embeds_all={"action": action_embeds},
            time_cond=time_cond,
            kv_caches=kv_caches,
            cache_mode="append_non_active",
        )["action"]
        action_vel = self.action_decoder(action_embeds)
        final_action = action + (1 - t) * action_vel
        with torch.no_grad():
            error_term = (target_action - final_action.detach()) * inpaint_attention[None, :, None]  # (bsz, horizon_steps, action_dim)
            error_term = error_term
        error = torch.dot(error_term.flatten(), final_action.flatten())

        action_grad = torch.autograd.grad(error, action)[0]
        coef = min(beta, (t ** 2 + (1 - t) ** 2) / (t * (1 - t))) if t > 0 and t < 1 else beta
        action = action + delta_t * (action_vel + coef * action_grad)
        t += delta_t
        return action, t

    def guided_inference(
        self,
        previous_action: torch.FloatTensor,
        delay: int,
        execution_horizon: int,
        beta: float,
        input_ids: torch.LongTensor,
        pointcloud: torch.FloatTensor,
        pcd_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Implementation of Real-Time Execution of Action Chunking Flow Policies (https://arxiv.org/pdf/2506.07339)
        """
        dtype, device = pointcloud.dtype, pointcloud.device
        bsz = pointcloud.size(0)
        # expand previous action to proper length
        with torch.no_grad():

            kv_caches = self.joint_model.build_mixture_caches()

            # merge the text tokens and the pcd tokens
            inputs_embeds = self._forward_pointcloud_and_text_embedding(input_ids, pointcloud)

            # proprio
            proprio_embeds = self.proprio_encoder(proprios)

            # forward pass thru the vlm and proprio, cache the kv
            _, kv_caches = self.joint_model(
                attention_mask=pcd_text_proprio_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "proprio": proprio_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds,
                    "proprio": proprio_embeds,
                },
                kv_caches=kv_caches,
                return_caches=True,
            )

        target_action = torch.zeros(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )
        target_action[:, : previous_action.size(1), :] = previous_action

        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype, requires_grad=True
        )

        inpaint_attention = torch.zeros((self.horizon_steps), device=device)
        idx = torch.arange(self.horizon_steps, device=device)
        c = (self.horizon_steps - execution_horizon - idx) / (self.horizon_steps - execution_horizon - delay + 1)
        exp = c * (torch.exp(c) - 1) / (np.e - 1)
        inpaint_attention[idx < self.horizon_steps - execution_horizon] = exp[idx < self.horizon_steps - execution_horizon]
        inpaint_attention[idx < delay] = 1.0

        for _ in range(self.num_inference_steps):
            action, t = self.guided_inference_iter(
                action,
                t,
                target_action,
                delta_t,
                beta,
                inpaint_attention,
                action_mask,
                action_position_ids,
                kv_caches,
            )
            

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    def infer_action_naive(
        self,
        input_ids: torch.LongTensor,
        pointcloud: torch.FloatTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pointcloud.dtype, pointcloud.device
        bsz = pointcloud.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the pcd tokens
        inputs_embeds = self._forward_pointcloud_and_text_embedding(input_ids, pointcloud)

        # encode proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- run vlm in each step, which is unnecessary
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "proprio": proprio_position_ids,
                    "action": action_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds.clone(),  # clone needed due to modified in-place
                    "proprio": proprio_embeds.clone(),
                    "action": action_embeds,
                },
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="no_append",  # no new tokens
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    def infer_text(
        self,
        input_ids: torch.LongTensor,
        pointcloud: torch.FloatTensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        q_len = input_ids.size(1)

        # text tokens + pcd tokens
        inputs_embeds = self._forward_pointcloud_and_text_embedding(input_ids, pointcloud)

        # build causal mask and position ids for text
        (
            causal_mask,
            position_ids,
        ) = self.build_causal_mask_and_position_ids_for_text(
            q_len, attention_mask, kv_cache
        )

        hidden_states = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={"vlm": position_ids},
            embeds_all={"vlm": inputs_embeds},
            kv_caches={"vlm": kv_cache},
            cache_mode="append",  # new tokens for the active mixture
            final_layer_post_attn_skip_names=[],  # do not skip vlm last layer
        )["vlm"]
        logits = self.lm_head(hidden_states)
        output = {
            "logits": logits,
        }
        if kv_cache is not None:
            output["kv_cache"] = kv_cache
        return output

    # ---------- Flow matching training ----------#

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def forward(
        self,
        input_ids: torch.LongTensor,
        pointcloud: torch.ByteTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        # text tokens + pcd tokens
        inputs_embeds = self._forward_pointcloud_and_text_embedding(input_ids, pointcloud)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)
        action_embeds = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
                "action": action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "action": action_embeds,
            },
            time_cond=time_cond,
            kv_caches={},  # no caching during training
        )["action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)


class PointCloudUniDexInference(PointCloudUniDex):
    def process(
        self,
        pointcloud: torch.FloatTensor,
        state: torch.FloatTensor,
        prompt: list, 
    ):
        model_inputs = self.processor(
            text=prompt,
            pcd=pointcloud,
        )
        input_ids = model_inputs["input_ids"].to(pointcloud.device)
        attention_mask = model_inputs["attention_mask"].to(pointcloud.device)
        pointcloud = model_inputs["pointcloud"]
        proprios = state
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.build_causal_mask_and_position_ids(attention_mask, pointcloud.dtype)
        )
        pcd_text_proprio_mask, action_mask = self.split_full_mask_into_submasks(
            causal_mask
        )
        return input_ids, pointcloud, pcd_text_proprio_mask, action_mask, vlm_position_ids, proprio_position_ids, action_position_ids, proprios

    def forward(
        self,
        previous_action: torch.FloatTensor,
        delay: int,
        execution_horizon: int,
        beta: float,
        pointcloud: torch.FloatTensor,
        state: torch.FloatTensor,
        prompt: list,
    ) -> torch.FloatTensor:
        with torch.no_grad():
            input_ids, pointcloud, pcd_text_proprio_mask, action_mask, vlm_position_ids, proprio_position_ids, action_position_ids, proprios = self.process(pointcloud, state, prompt)
        return super().guided_inference(
            previous_action,
            delay,
            execution_horizon,
            beta,
            input_ids,
            pointcloud,
            pcd_text_proprio_mask,
            action_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
            proprios,
        )
    
    @torch.inference_mode()
    def infer_action(
        self,
        pointcloud: torch.FloatTensor,
        state: torch.FloatTensor,
        prompt: list,
    ) -> torch.FloatTensor:
        input_ids, pointcloud, pcd_text_proprio_mask, action_mask, vlm_position_ids, proprio_position_ids, action_position_ids, proprios = self.process(pointcloud, state, prompt)
        return super().infer_action(
            input_ids,
            pointcloud,
            pcd_text_proprio_mask,
            action_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
            proprios,
        )
    
class PointCloudUniDexTrain(PointCloudUniDex):
    def forward(
            self,
            batch
    ):
        model_inputs = self.processor(
            text=batch["prompt"],
            pcd=batch["pointcloud"],
        )
        input_ids = model_inputs["input_ids"].to(batch["pointcloud"].device)
        attention_mask = model_inputs["attention_mask"].to(batch["pointcloud"].device)
        pointcloud = model_inputs["pointcloud"]
        actions = batch["action"]
        proprios = batch["state"]
        t = torch.rand(batch["action"].size(0), device=pointcloud.device, dtype=pointcloud.dtype)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.build_causal_mask_and_position_ids(attention_mask, pointcloud.dtype)
        )

        loss = super().forward(
            input_ids=input_ids,
            pointcloud=pointcloud,
            causal_mask=causal_mask,
            vlm_position_ids=vlm_position_ids,
            proprio_position_ids=proprio_position_ids,
            action_position_ids=action_position_ids,
            proprios=proprios,
            actions=actions,
            t=t,
        )

        return loss, dict()

    def infer_action(
            self,
            batch
    ):
        batch.pop("action", None)  # remove action from batch

        model_inputs = self.processor(
            text=batch["prompt"],
            pcd=batch["pointcloud"],
        )
        input_ids = model_inputs["input_ids"].to(batch["pointcloud"].device)
        attention_mask = model_inputs["attention_mask"].to(batch["pointcloud"].device)
        pointcloud = model_inputs["pointcloud"]
        proprios = batch["state"]
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.build_causal_mask_and_position_ids(attention_mask, pointcloud.dtype)
        )
        pcd_text_proprio_mask, action_mask = self.split_full_mask_into_submasks(
            causal_mask
        )

        return super().infer_action(
            input_ids=input_ids,
            pointcloud=pointcloud,
            pcd_text_proprio_mask=pcd_text_proprio_mask,
            action_mask=action_mask,
            vlm_position_ids=vlm_position_ids,
            proprio_position_ids=proprio_position_ids,
            action_position_ids=action_position_ids,
            proprios=proprios,
        )
