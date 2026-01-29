import math
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Trainer, MistralForCausalLM, GemmaForCausalLM, AutoTokenizer, Qwen2ForCausalLM
import torch.nn.functional as F
from tqdm import tqdm
import random
import warnings
from torch.optim import AdamW
from collections import deque
from transformers.modeling_outputs import CausalLMOutputWithPast
#
warnings.filterwarnings('ignore')
tqdm.pandas()

class FIFOBuffers:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def append(self, item):
        self.buffer.append(item)

    def get_stack(self):
        return torch.stack(list(self.buffer), dim=1)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class ConvAggregator(nn.Module):
    def __init__(self, n_layers, head_dim):
        super().__init__()
        self.depthwise = nn.Conv1d(head_dim, head_dim, kernel_size=3, padding=1, groups=head_dim)
        self.pointwise = nn.Conv1d(head_dim, head_dim, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv1d(head_dim, head_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = F.gelu(out)
        out = self.pointwise(out)
        gate = self.gate(x)
        out = out * gate
        out = out.mean(dim=-1)
        return out

class MultiLayerCLSProcessor_conv(nn.Module):
    def __init__(self, n_layers, dim, n_heads):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads
        assert dim % n_heads == 0
        self.head_dim = dim // n_heads

        self.aggregator_per_head = nn.ModuleList([
            ConvAggregator(n_layers, self.head_dim)
            for _ in range(n_heads)
        ])

        self.final_mlp = nn.Sequential(
            nn.Linear(n_heads * self.head_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim)
        )

    def forward(self, x):
        batch_size, n_layers, seq_len, hidden_size = x.size()
        x = x.view(batch_size, n_layers, self.n_heads, seq_len, self.head_dim)
        x = x.permute(0, 3, 2, 4, 1).contiguous()

        head_outputs = []
        for i in range(self.n_heads):
            head_input = x[:, :, i, :, :]
            B, L, D, N = head_input.size()
            head_input = head_input.view(B * L, D, N)
            out = self.aggregator_per_head[i](head_input)
            head_outputs.append(out)

        concatenated = torch.cat(head_outputs, dim=1)
        concatenated = concatenated.view(batch_size, seq_len, -1)
        final_output = self.final_mlp(concatenated)
        final_output = self.fallback_ln(final_output)
        return final_output

    def fallback_ln(self, x, eps=1e-3):
        mu = x.mean(dim=-1, keepdim=True)
        var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + eps * eps)
        std = torch.clamp(std, min=eps)
        return (x - mu) / std

class MultiLayerCLSProcessor_Linear(nn.Module):
    def __init__(self, n_layers, dim, n_heads):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads
        assert dim % n_heads == 0

        self.head_dim = dim // n_heads

        self.mlp_per_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_layers * self.head_dim, n_layers * self.head_dim * 2),
                nn.GELU(),
                nn.Linear(n_layers * self.head_dim * 2, self.head_dim)
            ) for _ in range(n_heads)
        ])

    def forward(self, x):
        batch_size, n_layers, seq_len, hidden_size = x.size()
        x_ = x.view(batch_size, n_layers, self.n_heads, seq_len, self.head_dim)
        x_ = x_.permute(0, 3, 2, 4, 1).contiguous()

        head_outputs = []
        for i in range(self.n_heads):
            head_input = x_[:, :, i, :, :]
            B_, L_, n_l, D_ = head_input.size()
            head_input_2d = head_input.contiguous().view(B_ * L_, n_l * D_)
            out = self.mlp_per_head[i](head_input_2d)
            head_outputs.append(out.squeeze(-1))

        concatenated_output = torch.cat(head_outputs, dim=-1)
        concatenated_output = concatenated_output.view(B_, L_, -1)
        final_output = self.fallback_ln(concatenated_output)
        return final_output

    def fallback_ln(self, x, eps=1e-3):
        mu = x.mean(dim=-1, keepdim=True)
        var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + eps * eps)
        std = torch.clamp(std, min=eps)
        return (x - mu) / std



class CustomPhiForCausalLM(Phi3ForCausalLM):
    def __init__(
        self,
        config,
        mi_max=0.05, mi_min=0.01, mi_warmup=0.3, mi_decay=0.99,
        ad_max=1.0, ad_min=0.01, ad_warmup=0.0, ad_decay=0.99,
        orth_max=0.05, orth_min=0.0001, orth_warmup=0.3, orth_decay=0.5
    ):
        config.output_hidden_states = True
        super(CustomPhiForCausalLM, self).__init__(config)

        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.global_step = 0

        self.shared_adapter = MultiLayerCLSProcessor_Linear(4, config.hidden_size, config.num_attention_heads)
        self.shared_adapter1 = MultiLayerCLSProcessor_conv(8, config.hidden_size, config.num_attention_heads)

        self.block_hidden_buffer = FIFOBuffers(max_size=4)
        self.first_level_buffer = FIFOBuffers(max_size=8)

        self.i = 0
        self.j = 0
        self.k = 0
        self.total_training_steps = None

        # Save dynamic weight scheduling parameters
        self.mi_max, self.mi_min, self.mi_warmup, self.mi_decay = mi_max, mi_min, mi_warmup, mi_decay
        self.ad_max, self.ad_min, self.ad_warmup, self.ad_decay = ad_max, ad_min, ad_warmup, ad_decay
        self.orth_max, self.orth_min, self.orth_warmup, self.orth_decay = orth_max, orth_min, orth_warmup, orth_decay

    def set_total_training_steps(self, total_steps):
        self.total_training_steps = total_steps

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, **kwargs):
        kwargs['output_hidden_states'] = True
        kwargs.setdefault('return_dict', True)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            **kwargs
        )

        hidden_states_final = outputs.hidden_states[-1]
        self._clear_all_buffers()
        hidden_source = outputs.hidden_states[:-1]

        for idx, hs in enumerate(hidden_source):
            self.i += 1
            self.block_hidden_buffer.append(hs.detach())

            if len(self.block_hidden_buffer) == 4 and self.i % 4 == 0:
                self.j += 1
                level1_output = self.shared_adapter(self.block_hidden_buffer.get_stack())
                self.first_level_buffer.append(level1_output)

                if len(self.first_level_buffer) == 8 and self.j % 8 == 0:
                    self.k += 1
                    level2_output = self.shared_adapter1(self.first_level_buffer.get_stack())
                    ccs_loss = self.centered_cosine_alignment_loss(level2_output.squeeze(1), hidden_states_final.detach())
                    orth_loss = self.orthogonality_loss(level2_output.squeeze(1), hidden_states_final.detach())
                    if random.random() >= 0.5:
                        level2_output = torch.zeros_like(level2_output)
                    gate = torch.sigmoid(self.gate_proj(hidden_states_final))

        self.global_step += 1

        mi_weight = self.cosine_bump(
            step=self.global_step,
            max_weight=self.mi_max,
            min_weight=self.mi_min,
            total_steps=self.total_training_steps,
            warmup_ratio=self.mi_warmup,
            decay_ratio=self.mi_decay,
        )

        ad_weight = self.cosine_bump(
            step=self.global_step,
            max_weight=self.ad_max,
            min_weight=self.ad_min,
            total_steps=self.total_training_steps,
            warmup_ratio=self.ad_warmup,
            decay_ratio=self.ad_decay,
        )

        orth_weight = self.cosine_bump(
            step=self.global_step,
            max_weight=self.orth_max,
            min_weight=self.orth_min,
            total_steps=self.total_training_steps,
            warmup_ratio=self.orth_warmup,
            decay_ratio=self.orth_decay,
        )

        hidden_states_fin = hidden_states_final + gate * level2_output * ad_weight
        mi_loss = self.cosine_sim_loss(hidden_states_final, hidden_states_fin.detach())
        logits = self.lm_head(hidden_states_fin)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss += 0.1 * ccs_loss + orth_loss * orth_weight + mi_loss * mi_weight
            print(f"CCS Loss: {ccs_loss.item():.4f}", f"Orth Loss: {orth_loss.item():.2f}", f"MI Loss: {mi_loss.item():.2f}")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def cosine_bump(self, step, max_weight, min_weight, total_steps=None, warmup_ratio=0.1, decay_ratio=0.6):
        warmup_steps = int(total_steps * warmup_ratio)
        decay_speed = int(total_steps * decay_ratio)

        if step < warmup_steps:
            return max_weight * step / warmup_steps
        else:
            decay = 0.5 * (1 + math.cos(min(1.0, (step - warmup_steps) / decay_speed) * math.pi))
            return (max_weight - min_weight) * decay + min_weight

    def _clear_all_buffers(self):
        self.block_hidden_buffer.clear()
        self.first_level_buffer.clear()
        self.i = 0
        self.j = 0
        self.k = 0

    def cosine_sim_loss(self, student_hidden, teacher_hidden):
        student_norm = torch.norm(student_hidden, dim=-1, keepdim=True).clamp(min=1e-3)
        teacher_norm = torch.norm(teacher_hidden, dim=-1, keepdim=True).clamp(min=1e-3)

        student_normalized = student_hidden / student_norm
        teacher_normalized = teacher_hidden / teacher_norm

        cosine_sim = torch.sum(student_normalized * teacher_normalized, dim=-1)
        cosine_sim = torch.where(torch.isnan(cosine_sim), torch.zeros_like(cosine_sim), cosine_sim)

        return -cosine_sim.mean()

    def centered_cosine_alignment_loss(self, adapter_output, hidden_states_final):
        adapter_centered = adapter_output - adapter_output.mean(dim=-1, keepdim=True)
        hidden_centered = hidden_states_final - hidden_states_final.mean(dim=-1, keepdim=True)
        cosine_sim = F.cosine_similarity(adapter_centered, hidden_centered, dim=-1)
        return 1.0 - cosine_sim.mean()

    def orthogonality_loss(self, aggregator_output, hidden_states_final):
        aggregator_norm = self.normalize(aggregator_output)
        hidden_norm = self.normalize(hidden_states_final)

        B, L, D = aggregator_norm.size()
        agg_flat = aggregator_norm.view(B * L, D)
        hid_flat = hidden_norm.view(B * L, D)
        dot = torch.sum(agg_flat * hid_flat, dim=-1)
        return torch.mean(dot ** 2)

    def variance_regularization(self, aggregator_output):
        var_aggregator = torch.var(aggregator_output, dim=1, unbiased=True)
        return -torch.mean(var_aggregator)

    def _normalize(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + 1e-6)
        return (x - mean) / std

    def normalize(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-3)
        return x / norm

    def generate(self, input_ids=None, past_key_values=None, **kwargs):
        return super().generate(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
