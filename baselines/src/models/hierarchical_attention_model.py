import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F

import math
from inspect import isfunction

# constants

MIN_EXPERT_CAPACITY = 4

# helper functions

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]


def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

# activations

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# expert class

class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out


class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance = None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # BALANCING LOSSES
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat
        
        # [batch, group, experts, expert_capacity]
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss

# plain mixture of experts

class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        self.num_experts = num_experts

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef

# 2-level heirarchical mixture of experts

class HeirarchicalMoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = (4, 4),
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of heirarchy for experts allowed for now'
        num_experts_outer, num_experts_inner = num_experts
        self.num_experts_outer = num_experts_outer
        self.num_experts_inner = num_experts_inner

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}

        self.gate_outer = Top2Gating(dim, num_gates = num_experts_outer, **gating_kwargs)
        self.gate_inner = Top2Gating(dim, num_gates = num_experts_inner, outer_expert_dims = (num_experts_outer,), **gating_kwargs)

        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        # we construct an "importance" Tensor for the inputs to the second-level
        # gating.  The importance of an input is 1.0 if it represents the
        # first-choice expert-group and 0.5 if it represents the second-choice expert
        # group.  This is used by the second-level gating.
        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance = importance)
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # NOW COMBINE EXPERT OUTPUTS (reversing everything we have done)
        # expert_output has shape [y0, x1, h, d, n]

        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
        return output, (loss_outer + loss_inner) * self.loss_coef


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Attention'
        self.masks = []
        self.sparsity_ratios = []
        self.params = {}
        self.layer_counter = 0

    def info(self):
        return {
            "name": self.name,
            "params": self.params,
            "masks": self.masks,
            "sparsity": self.calculate_average_sparsity_ratio(self.sparsity_ratios)
        }

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def maybe_save_mask(self, attention_mask, modulo_layers=8):
        if self.layer_counter % modulo_layers == 0:
            self.masks.append(attention_mask[0, 0].clone().cpu().numpy())

        self.layer_counter += 1

    def forward(self, queries, keys, values, *args, **kwargs):
        """Base forward method for prefilling attention patterns.
        
        Args:
            queries: (batch_size, num_heads, seq_len, head_dim)
            keys: (batch_size, num_heads, seq_len, head_dim)
            values: (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            attention_output: (batch_size, num_heads, seq_len, head_dim)
        """
        raise NotImplementedError

    def generation_forward(self, prefilling_queries, prefilling_keys, prefilling_values,
                         generation_queries, generation_keys, generation_values, *args, **kwargs):
        """Base forward method for generation attention patterns.
        
        Args:
            prefilling_queries: Queries from prefilling stage
            prefilling_keys: Keys from prefilling stage
            prefilling_values: Values from prefilling stage
            generation_queries: New queries for generation
            generation_keys: New keys for generation
            generation_values: New values for generation
            
        Returns:
            attention_output: Output for generation tokens
        """
        raise NotImplementedError

    @staticmethod
    def attention(queries, keys, values, attention_mask, return_attention_scores=False):
        """Standard attention computation."""
        attention_weights = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(queries.size(-1))
        attention_weights += attention_mask.to(queries.dtype) * torch.finfo(queries.dtype).min
        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attention_output = torch.matmul(attention_weights, values)
        if return_attention_scores:
            return attention_output, attention_weights
        return attention_output
    
    @staticmethod
    def get_causal_mask(seq_len, device):
        """Creates a causal mask where future tokens cannot attend to past tokens.
        
        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on
            
        Returns:
            mask: Boolean tensor of shape (1, 1, seq_len, seq_len) where True/1 indicates
                 that position (i,j) should be masked (set to -inf before softmax)
        """
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def get_local_mask(seq_len, window_size, device):
        """Creates a local attention mask where tokens can only attend to nearby tokens
        within a fixed window size, plus the causal constraint.
        
        Args:
            seq_len: Length of the sequence
            window_size: Size of the local attention window including current token
            device: Device to create the mask on
            
        Returns:
            mask: Boolean tensor of shape (1, 1, seq_len, seq_len) where True/1 indicates
                 that position (i,j) should be masked (set to -inf before softmax)
        """
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        mask = torch.triu(mask, diagonal=-(window_size-1))
        mask = torch.tril(mask, diagonal=0)
        mask = (~mask)
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def get_generation_mask(gen_len, prefill_len, device):
        """Creates a mask that allows generation tokens to:
        1. Attend to all prefilling tokens
        2. Attend causally to other generation tokens
        
        Args:
            gen_len: Number of tokens being generated
            prefill_len: Number of tokens in the prefill context
            device: Device to create the mask on
            
        Returns:
            mask: Boolean tensor of shape (1, 1, gen_len, prefill_len + gen_len)
                 where True indicates positions that should be masked
        """
        return torch.cat([
            torch.zeros((1, 1, gen_len, prefill_len), dtype=torch.bool, device=device),
            Attention.get_causal_mask(gen_len, device)
        ], dim=-1)

    @staticmethod
    def calculate_sparsity_ratio(mask):
        """Calculates the sparsity ratio of an attention mask.

        This method computes what fraction of the possible attention connections are masked 
        assuming that attention is causal, i.e., that tokens cannot attend to tokens before them.
        A higher ratio means more sparse attention. Asummes batch_size = 1.

        Args:
            mask: Boolean tensor of shape (batch_size, num_heads, queries_len, keys_len) where
                 True/1 indicates masked (disabled) attention connections

        Returns:
            float: The sparsity ratio between 0 and 1, where:
                  0 means all possible connections are enabled (dense attention)
                  1 means all possible connections are masked (completely sparse)
        """

        _, _, queries_len, keys_len = mask.shape

        if queries_len != keys_len:
            prefill_length = keys_len - queries_len
            total_connections = queries_len * (queries_len + 1) // 2 + prefill_length * queries_len
        else:
            total_connections = queries_len * (queries_len + 1) // 2

        connections_per_head = (~mask).long().sum(dim=(-1, -2))
        non_masked_ratio = (connections_per_head.float() / total_connections).mean(dim=-1).item()
        sparsity_ratio = 1 - non_masked_ratio

        return sparsity_ratio
    
    @staticmethod
    def calculate_average_sparsity_ratio(sparsity_ratios):
        return sum(sparsity_ratios) / len(sparsity_ratios) if sparsity_ratios else 0
    
class DynamicAttentionWindow:
    def __init__(self, min_window=32, max_window=512, base_window=128):
        self.min_window = min_window
        self.max_window = max_window
        self.base_window = base_window
        
    def get_window_size(self, input_length: int, complexity_score: float) -> int:
        # 根据输入长度和复杂度动态计算窗口大小
        base_size = min(self.base_window * (1 + complexity_score), self.max_window)
        return max(self.min_window, min(base_size, input_length // 4))

class HierarchicalAttention(nn.Module):
    def __init__(self, config, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.sentence_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        self.paragraph_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        self.document_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 句子级注意力
        sentence_output, sentence_attn = self.sentence_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask,
            need_weights=output_attentions
        )
        sentence_output = self.layer_norm1(hidden_states + sentence_output)
        
        # 段落级注意力
        paragraph_output, paragraph_attn = self.paragraph_attention(
            sentence_output, sentence_output, sentence_output,
            key_padding_mask=attention_mask,
            need_weights=output_attentions
        )
        paragraph_output = self.layer_norm2(sentence_output + paragraph_output)
        
        # 文档级注意力
        document_output, document_attn = self.document_attention(
            paragraph_output, paragraph_output, paragraph_output,
            key_padding_mask=attention_mask,
            need_weights=output_attentions
        )
        document_output = self.layer_norm3(paragraph_output + document_output)
        
        if output_attentions:
            return document_output, (sentence_attn, paragraph_attn, document_attn)
        return document_output, None

class HierarchicalAttentionConfig(PretrainedConfig):
    model_type = "hierarchical_attention"
    
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings

class HierarchicalAttentionModel(PreTrainedModel):
    config_class = HierarchicalAttentionConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # 初始化动态注意力窗口
        self.dynamic_window = DynamicAttentionWindow(
            min_window=32,
            max_window=512,
            base_window=128
        )
        
        # 初始化分层注意力层
        self.hierarchical_attention = HierarchicalAttention(
            config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads
        )
        
        # 其他必要的层
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ):
        # 获取输入长度
        input_length = input_ids.size(1)
        
        # 计算复杂度分数（这里使用一个简单的启发式方法）
        complexity_score = torch.mean(attention_mask.float()) if attention_mask is not None else 0.5
        
        # 获取动态窗口大小
        window_size = self.dynamic_window.get_window_size(input_length, complexity_score)
        
        # 获取嵌入
        embeddings = self.embedding(input_ids)
        position_ids = torch.arange(0, input_length, dtype=torch.long, device=input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = embeddings + position_embeddings
        
        # 应用分层注意力
        outputs = self.hierarchical_attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        
        return outputs 