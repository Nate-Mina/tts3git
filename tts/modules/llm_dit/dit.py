# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc

import torch
from torch import nn

from tts.modules.llm_dit.cfm import ConditionalFlowMatcher
from tts.modules.ar_dur.commons.layers import Embedding
from tts.modules.ar_dur.commons.nar_tts_modules import PosEmb
from tts.modules.ar_dur.commons.rel_transformer import RelTransformerEncoder
from tts.modules.ar_dur.ar_dur_predictor import expand_states
from tts.modules.llm_dit.transformer import Transformer
from tts.modules.llm_dit.time_embedding import TimestepEmbedding

# Print the sequential-CFG notice once per process, not once per denoise step.
_seq_cfg_warned = [False]

# Above this many frames, the batched (3x) CFG stack is too large for a 12 GB
# card. Measured: comfortable to ~2100 frames, marginal beyond. The sequential
# path uses ~1/3 the activation memory.
_SEQ_CFG_SEQ_THRESHOLD = 2100


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Hparams
        # cond dim
        self.local_cond_dim = 512
        self.ctx_mask_dim = 16
        self.in_channels = 32
        self.out_channels = 32
        # LLM
        self.encoder_dim = 1024
        self.encoder_n_layers = 24
        self.encoder_n_heads = 16
        self.max_seq_len = 16384
        self.multiple_of = 256

        self.ctx_mask_proj = nn.Linear(1, self.ctx_mask_dim)
        self.local_cond_project = nn.Linear(
            self.out_channels + self.ctx_mask_dim, self.local_cond_dim)

        self.encoder = Transformer(self.encoder_n_layers, self.encoder_dim, self.encoder_n_heads, self.max_seq_len)

        self.x_prenet = nn.Linear(self.in_channels, self.encoder_dim)
        self.prenet = nn.Linear(self.local_cond_dim, self.encoder_dim)
        self.postnet = nn.Linear(self.encoder_dim, self.out_channels)
  
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        # The implementation of TimestepEmbedding is a modified version from F5-TTS (https://github.com/SWivid/F5-TTS), 
        # which is licensed under the MIT License.
        self.f5_time_embed = TimestepEmbedding(self.encoder_dim)

        # text encoder
        self.ph_encoder = RelTransformerEncoder(
            302, self.encoder_dim, self.encoder_dim,
            self.encoder_dim * 2, 4, 6,
            3, 0.0, prenet=True, pre_ln=True)
        self.tone_embed = Embedding(32, self.encoder_dim, padding_idx=0)
        self.ph_pos_embed = PosEmb(self.encoder_dim)
        self.ling_pre_net = torch.nn.Sequential(*[
            torch.nn.Conv1d(self.encoder_dim, self.encoder_dim, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate([2, 2])
        ])
    
    def forward(self, inputs, sigmas=None, x_noisy=None):
        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        """ local conditioning (prompt_latent + spk_embed) """
        ctx_mask_emb = self.ctx_mask_proj(ctx_mask)
        # ctx_feature = ctx_feature * (1 - inputs["spk_cfg_mask"][:, :, None])
        local_cond = torch.cat([ctx_feature, ctx_mask_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)

        """ diffusion target latent """
        x = inputs['lat']
    
        # Here, x is x1 in CFM
        x0 = torch.randn_like(x)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
        
        # define noisy_input and target
        t = t.bfloat16()
        x_noisy = (xt * (1 - ctx_mask)).bfloat16()
        target = ut

        # concat condition.
        x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
        x_ling = self.ling_pre_net(expand_states(x_ling, inputs['mel2ph']).transpose(1, 2)).transpose(1, 2)
        x_noisy = self.x_prenet(x_noisy) + self.prenet(local_cond) + x_ling
        encoder_out = self.encoder(x_noisy, self.f5_time_embed(t), attn_mask=inputs["text_mel_mask"], do_checkpoint=False)
        pred = self.postnet(encoder_out)

        return pred, target
    
    def forward_ling_encoder(self, txt_tokens, tone_tokens):
        ph_tokens = txt_tokens
        ph_nonpadding = (ph_tokens > 0).float()[:, :, None]  # [B, T_phone, 1]

        # enc_ph
        ph_enc_oembed = self.tone_embed(tone_tokens)
        ph_enc_oembed = ph_enc_oembed + self.ph_pos_embed(
            torch.arange(0, ph_tokens.shape[1])[None,].to(ph_tokens.device))
        ph_enc_oembed = ph_enc_oembed
        ph_enc_oembed = ph_enc_oembed * ph_nonpadding
        x_ling = self.ph_encoder(ph_tokens, other_embeds=ph_enc_oembed) * ph_nonpadding
        return x_ling

    def _forward(self, x, local_cond, x_ling, timesteps, ctx_mask, dur=None, seq_cfg_w=[1.0,1.0]):
        """ When we use torchdiffeq, we need to include the CFG process inside _forward() """
        x = x * (1 - ctx_mask)
        x = self.x_prenet(x) + self.prenet(local_cond) + x_ling
        pred_v = self.encoder(x, self.f5_time_embed(timesteps), attn_mask=torch.ones((x.size(0), x.size(1)), device=x.device))
        pred = self.postnet(pred_v)

        """ Perform multi-cond CFG """
        cond_spk_txt, cond_txt, uncond = pred.chunk(3)
        pred = uncond + seq_cfg_w[0] * (cond_txt - uncond) + seq_cfg_w[1] * (cond_spk_txt - cond_txt)
        return pred

    def _forward_seq_cfg(self, x, local_cond, x_ling, timesteps, ctx_mask, dur=None, seq_cfg_w=[1.0, 1.0]):
        """Identical maths to _forward, with the CFG batch run one slice at a time.

        _forward concatenates the three CFG conditions (spk+txt, txt, uncond) into
        a single batch of 3, which makes every activation in the 24-layer stack
        3x larger than it needs to be. At seq 4400 that stack is ~2.4 GB and is
        what actually exhausts a 12 GB card -- not the attention score matrix.

        Every op in this path (attention over the sequence axis, LayerNorm,
        Linear, the postnet) is independent per batch element, so slicing the
        batch is exact. It trades compute (three passes instead of one) for peak
        memory (a third of the activations). scripts/dml_diff_probe.py asserts
        the two agree.
        """
        t_emb = self.f5_time_embed(timesteps)
        preds = []
        # x is the shared noisy latent (batch 1); the 3 CFG conditions live in
        # local_cond / x_ling (batch 3). Loop over the condition axis, not x.
        for i in range(local_cond.size(0)):
            xi = x * (1 - ctx_mask[i:i + 1])
            xi = self.x_prenet(xi) + self.prenet(local_cond[i:i + 1]) + x_ling[i:i + 1]
            pred_v = self.encoder(
                xi, t_emb,
                attn_mask=torch.ones((xi.size(0), xi.size(1)), device=xi.device))
            preds.append(self.postnet(pred_v))

        if len(preds) != 3:
            raise RuntimeError(
                f"sequential CFG produced {len(preds)} slices, expected 3 "
                f"(a slice failed -- likely OOM even on the sequential path)")
        cond_spk_txt, cond_txt, uncond = preds
        return uncond + seq_cfg_w[0] * (cond_txt - uncond) + seq_cfg_w[1] * (cond_spk_txt - cond_txt)

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.0, 1.0], **kwargs):
        # txt embedding
        x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
        x_ling = self.ling_pre_net(expand_states(x_ling, inputs['dur']).transpose(1, 2)).transpose(1, 2)

        # speaker embedding
        ctx_feature = inputs['lat_ctx']
        ctx_feature[1:, :, :] = 0 # prefix spk cfg
        ctx_mask_emb = self.ctx_mask_proj(inputs['ctx_mask'])

        # local conditioning.
        local_cond = torch.cat([ctx_feature, ctx_mask_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)
        
        ''' Euler ODE solver '''
        bsz, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        # Sway sampling from F5-TTS (https://github.com/SWivid/F5-TTS), 
        # which is licensed under the MIT License.
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1, device=device, dtype=x_ling.dtype)
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)
        
        # AMO sampling implementation for "AMO Sampler: Enhancing Text Rendering with Overshooting" (https://arxiv.org/pdf/2411.19415)
        def amo_sampling(z_t, t, t_next, v):
            # Upcast to avoid precision issues when computing prev_sample
            z_t = z_t.to(torch.float32)

            # Constant definition in Algorithm 1
            s = t_next
            c = 3

            # Line 7 in Algorithm 1
            o = min(t_next + c * (t_next - t), 1)
            pred_z_o = z_t + (o - t) * v

            # Line 11 in Algorithm 1
            a = s / o
            b = ((1 - s) ** 2 - (a * (1 - o)) ** 2) ** 0.5
            noise_i = torch.randn(size=z_t.shape, device=z_t.device)
            z_t_next = a * pred_z_o + b * noise_i
            return z_t_next.to(v.dtype)

        x = torch.randn([1, frm_len, self.out_channels], device=device)
        # Choose the CFG path UP FRONT rather than discovering it by failing.
        #
        # A failed batched attempt is expensive in the one resource that matters:
        # the exception's traceback pins the partially-built activation stack, so
        # `gc.collect()` does not free it and the fallback then runs on an
        # already-full card -- which is exactly how the first version of this
        # fallback died. Deciding from the sequence length costs nothing and
        # never wastes an attempt.
        #
        # Threshold from measurement on a 12 GB card: the batched (3x) stack is
        # comfortable to ~2100 frames and marginal beyond; the sequential stack
        # is ~1/3 the size.
        seq_len = frm_len
        use_seq_cfg = seq_len > _SEQ_CFG_SEQ_THRESHOLD
        if use_seq_cfg and not _seq_cfg_warned[0]:
            print(f"[MegaTTS3] seq={seq_len} > {_SEQ_CFG_SEQ_THRESHOLD}: using "
                  f"sequential CFG passes (1/3 the activation memory, ~3x compute).")
            _seq_cfg_warned[0] = True

        for step_index in range(timesteps):
            x = x.to(torch.float32)
            sigma = t_schedule[step_index].to(x_ling.dtype)
            sigma_next = t_schedule[step_index + 1]
            if use_seq_cfg:
                model_out = self._forward_seq_cfg(x, local_cond, x_ling, timesteps=sigma.unsqueeze(0), ctx_mask=inputs['ctx_mask'], dur=inputs['dur'], seq_cfg_w=seq_cfg_w)
            else:
                try:
                    model_out = self._forward(torch.cat([x] * bsz), local_cond, x_ling, timesteps=sigma.unsqueeze(0), ctx_mask=inputs['ctx_mask'], dur=inputs['dur'], seq_cfg_w=seq_cfg_w)
                except RuntimeError as e:
                    if "not enough GPU video memory" not in str(e):
                        raise
                    # Drop the traceback before collecting: it holds frames that
                    # reference the half-built stack, so gc alone frees nothing.
                    e = None
                    gc.collect()
                    use_seq_cfg = True
                    if not _seq_cfg_warned[0]:
                        print("[MegaTTS3] Batched CFG exhausted VRAM despite the "
                              "threshold; switching to sequential CFG passes.")
                        _seq_cfg_warned[0] = True
                    model_out = self._forward_seq_cfg(x, local_cond, x_ling, timesteps=sigma.unsqueeze(0), ctx_mask=inputs['ctx_mask'], dur=inputs['dur'], seq_cfg_w=seq_cfg_w)
            x = amo_sampling(x, sigma, sigma_next, model_out)
            # Cast sample back to model compatible dtype
            x = x.to(model_out.dtype)

        return x
