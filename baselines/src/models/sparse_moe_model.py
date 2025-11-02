import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, List
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = torch.tril(torch.ones([n, n]))
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones([n, n]), ctx)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError('Not yet implemented')
    b = torch.reshape(b, [1, 1, n, n])
    return b

def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    n, t, embd = x.size()
    x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    x = torch.transpose(x, 0, 2, 1, 3)
    x = torch.reshape(x, [n, t, embd])
    return x

def split_heads(x, n):
    return torch.transpose(split_states(x, n), 0, 2, 1, 3)

def merge_heads(x):
    return merge_states(torch.transpose(x, 0, 2, 1, 3))

def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return torch.reshape(x, new_x_shape)


    return torch.reshape(x, new_x_shape)

def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = x.size()
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return torch.reshape(x, new_x_shape)

def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = k.size()[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx).float()
    w = torch.matmul(q, k.transpose(-2, -1))
    scale_amount = 1.0 / np.sqrt(q.size()[-1])
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = F.softmax(w, dim=-1)
    a = torch.matmul(w, v)
    a = merge_heads(a)
    return a

def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=32, num_verts=None, vertsize=None):
    n_ctx = q.size()[1]
    if attn_mode == 'strided':
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = q.size()[-1] // heads
    scale_amount = 1.0 / np.sqrt(n_state)
    w = torch.matmul(q, k.transpose(-2, -1))
    w = F.softmax(w * scale_amount, dim=-1)
    a = torch.matmul(w, v)
    if attn_mode == 'strided':
        n, t, embd = a.size()
        bT_ctx = n_ctx // local_attn_ctx
        a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        a = torch.transpose(a, 0, 2, 1, 3)
        a = torch.reshape(a, [n, t, embd])
    return a

class SparseAttentionn(nn.Module):
    def __init__(self, heads, attn_mode, local_attn_ctx=None, blocksize=32):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize

    def forward(self, q, k, v):
        return blocksparse_attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx)
import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import einsum, rearrange
from einops.layers.torch import EinMix as Mix, Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# start accumulating some types of compression networks

class ConvLinearCompress(Module):
    """
    used successfully in an old google brain paper, https://github.com/lucidrains/memory-efficient-attention-pytorch
    grouped convolutions so each head get its own parameters
    """

    def __init__(
        self,
        heads,
        dim_head,
        compress_window_size
    ):
        super().__init__()
        self.heads = heads
        self.conv = nn.Conv1d(heads * dim_head, heads * dim_head, compress_window_size, stride = compress_window_size, groups = heads)

    def forward(
        self,
        kv # Float['b h w n d']
    ):

        kv = rearrange(kv, 'b h w n d -> b (h d) (w n)')

        compressed = self.conv(kv)

        return rearrange(compressed, 'b (h d) n -> b h n d', h = self.heads)

# attention pool used by enformer, deepmind's genetic attention network

class AttentionPool(Module):
    def __init__(
        self,
        dim_head,
        compress_window_size
    ):
        super().__init__()
        self.to_attn_logits = nn.Linear(dim_head, dim_head, bias = False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim_head))

    def forward(
        self,
        kv
    ):

        attn_logits = self.to_attn_logits(kv)

        attn = attn_logits.softmax(dim = -2)

        compressed = einsum(kv, attn, 'b h w n d, b h w n d -> b h w d')

        return compressed

# mlp per head

class GroupedMLP(Module):
    def __init__(
        self,
        dim_head,
        compress_window_size,
        heads,
        expand_factor = 1.,
    ):
        super().__init__()

        dim = dim_head * compress_window_size
        dim_hidden = int(dim * expand_factor)
        dim_out = dim_head

        self.net = nn.Sequential(
            Mix('b h w i -> b h w o', weight_shape = 'h i o', bias_shape = 'h o', h = heads, i = dim, o = dim_hidden),
            nn.ReLU(),
            Mix('b h w i -> b h w o', weight_shape = 'h i o', bias_shape = 'h o', h = heads, i = dim_hidden, o = dim_out),
        )

    def forward(
        self,
        kv
    ):
        kv = rearrange(kv, 'b h w n d -> b h w (n d)')

        compressed = self.net(kv)

        return compressed

# single projection "mlp"

class SingleProjection(Module):
    def __init__(
        self,
        dim_head,
        compress_window_size,
        heads = 1
    ):
        super().__init__()
        dim = dim_head * compress_window_size
        dim_out = dim_head

        is_grouped = heads > 1

        if not is_grouped:
            self.compress = nn.Sequential(
                Rearrange('b h w n d -> b h w (n d)'),
                nn.Linear(dim, dim_out, bias = False)
            )
        else:
            self.compress = Mix(
                'b h w n i -> b h w o',
                weight_shape = 'h i o',
                h = heads,
                i = dim_head,
                o = dim_head
            )

    def forward(
        self,
        kv
    ):
        return self.compress(kv)

# simple transformer compressor, pull requested by Eric Pasewark

class SimpleMultiheadSelfAttention(Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout_p = dropout

    def forward(self, x):
        B, L, D = x.shape 
        q = self.q_proj(x)  # (B, L, D)
        k = self.k_proj(x)  # (B, L, D)
        v = self.v_proj(x)  # (B, L, D)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # -> (B, num_heads, L, head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # -> (B, num_heads, L, head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # -> (B, num_heads, L, head_dim)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(attn_out)

class SimpleTransformerFeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        """Two-layer feed-forward network with GELU activation."""
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out

class SimpleTransformerLayer(Module):
    def __init__(self, dim, num_heads, ff_hidden_dim=None, dropout=0.0):
        """Single Transformer layer: RMSNorm + Multi-head attention + RMSNorm + FeedForward."""
        super().__init__()
        if ff_hidden_dim is None:
            ff_hidden_dim = dim * 4
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.attn = SimpleMultiheadSelfAttention(dim, num_heads, dropout=dropout)
        self.ff   = SimpleTransformerFeedForward(dim, ff_hidden_dim, dropout=dropout)

    def forward(self, x):
        a = self.attn(self.norm1(x))
        x = x + a
        f = self.ff(self.norm2(x))
        x = x + f
        return x

class CompressTransformer(Module):
    def __init__(self, num_layers, dim, num_heads, ff_hidden_dim=None, dropout=0.0):
        """
        Stacked Transformer encoder layers.
        Args:
          num_layers: number of TransformerLayer to stack.
          dim: hidden dimension of the model (and input embeddings).
          num_heads: number of attention heads.
          ff_hidden_dim: hidden size of feed-forward network (defaults to 4*dim).
          dropout: dropout rate for attention weights and feed-forward (if any).
        """
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(dim, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x):
        # x shape: [b, h, w, n, d]
        b, h, w, n, d = x.shape

        # Flatten so each window is treated like a batch element: [b*w, n, h*d]
        inp = x.permute(0, 2, 3, 1, 4).contiguous()
        inp = inp.view(b*w, n, h*d)

        for i in range(self.num_layers - 1):
            inp = self.layers[i](inp)

        last_layer = self.layers[-1]

        a = last_layer.attn(last_layer.norm1(inp))
        inp = inp + a

        # Extract the last token along the 'n' dimension
        last_token = inp[:, -1].unsqueeze(1)  # (bw, 1, hd)

        normed = last_layer.norm2(last_token)
        ff_out = last_layer.ff(normed)
        last_token = last_token + ff_out

        last_token = last_token.squeeze(1).view(b, w, h, d).permute(0, 2, 1, 3)

        return last_token

class SparseAttention(nn.Module):
    def __init__(self, config, num_heads: int, window_size: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = config.hidden_size // num_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        
        # 投影到查询、键、值
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 计算稀疏注意力分数
        attn_scores = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=hidden_states.device)
        
        # 使用滑动窗口计算局部注意力
        for i in range(0, seq_len, self.window_size):
            end = min(i + self.window_size, seq_len)
            local_q = q[:, i:end]
            local_k = k[:, i:end]
            local_scores = torch.matmul(local_q, local_k.transpose(-2, -1)) / self.scale
            attn_scores[:, :, i:end, i:end] = local_scores
            
            # 添加全局token的注意力
            if i > 0:
                global_scores = torch.matmul(local_q, k[:, :i].transpose(-2, -1)) / self.scale
                attn_scores[:, :, i:end, :i] = global_scores
                
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.out_proj(context)

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.dense2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states

class MoELayer(nn.Module):
    def __init__(self, config, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([Expert(config) for _ in range(num_experts)])
        self.gate = nn.Linear(config.hidden_size, num_experts)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # 计算门控分数
        gate_scores = self.gate(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # 选择top-k专家
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = F.softmax(top_k_scores, dim=-1)
        
        # 初始化输出
        output = torch.zeros_like(hidden_states)
        
        # 计算每个专家的输出
        for i in range(self.num_experts):
            # 获取当前专家的mask
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.any():
                # 获取当前专家的输入
                expert_input = hidden_states[expert_mask]
                # 计算专家输出
                expert_output = self.experts[i](expert_input)
                # 计算加权和
                expert_weights = top_k_scores[expert_mask, :, (top_k_indices[expert_mask] == i).nonzero()[:, 1]]
                output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)
                
        return output

class SparseMoEConfig(PretrainedConfig):
    model_type = "sparse_moe"
    
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        num_experts=8,
        top_k=2,
        attention_window=32,
        hidden_dropout_prob=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.attention_window = attention_window
        self.hidden_dropout_prob = hidden_dropout_prob

class SparseMoEModel(PreTrainedModel):
    config_class = SparseMoEConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # 初始化稀疏注意力层
        self.attention = SparseAttention(
            config,
            num_heads=config.num_attention_heads,
            window_size=config.attention_window
        )
        
        # 初始化MoE层
        self.moe = MoELayer(
            config,
            num_experts=config.num_experts,
            top_k=config.top_k
        )
        
        # 其他必要的层
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ):
        # 获取输入长度
        input_length = input_ids.size(1)
        
        # 获取嵌入
        embeddings = self.embedding(input_ids)
        position_ids = torch.arange(0, input_length, dtype=torch.long, device=input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = embeddings + position_embeddings
        
        # 应用稀疏注意力
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(hidden_states + attention_output)
        
        # 应用MoE层
        moe_output = self.moe(attention_output)
        moe_output = self.dropout(moe_output)
        output = self.layer_norm(attention_output + moe_output)
        
        return output 