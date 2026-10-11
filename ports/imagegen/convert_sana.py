"""SANA-Sprint 0.6B (1024 px) to Core ML for the Neural Engine.

    python convert_sana.py <diffusers repo dir> <out dir> transformer
    python convert_sana.py <diffusers repo dir> <out dir> decoder_tail

The transformer's blocks are one program for the Neural Engine, taking the
scaled latent, the projected prompt (300 positions), an additive mask for
them, and the timestep's modulation vectors; the sampler arithmetic around
it stays in Swift (SanaPipeline.swift). `export_sana_mlx.py` writes what
runs in MLX instead: Gemma, the caption projection, the modulation for the
fixed two-step schedule, and the DC-AE decoder's first stages.

The DC-AE decoder is split. Its first three stages (32 to 128 px) have
linear attention, whose sums reach 400,000 and whose matmul the Neural
Engine gets wrong outright in float16, so they run in MLX on the GPU
(SanaDecoder.swift). The last three (256 to 1,024 px) are plain
convolution blocks — 90% of the decoder's arithmetic — and run here, on
the Neural Engine, from stage three's output.

SiLU, rewritten for the Neural Engine: its built-in sigmoid is a coarse
approximation (0.25% error), and the Mix-FFN multiplies one SiLU by another
branch, which made that 2.6% in every block — cosine 0.979 for the whole
transformer against float32, where the GPU in float16 gives 0.99995.
Written as x / (1 + exp(-x)) the same block is 0.11%.

DC-AE in float16: its last stages carry values near 1,850, and its RMSNorms
square them — float16 overflows, and the picture came out as noise (cosine
0.57 on every compute unit; 1.000000 in float32). A norm doesn't depend on
its input's scale, so each pixel is divided by its largest channel first
(at least 1) and the epsilon by that squared: the same result, and the
squares stay at most 1 whatever the prompt.

Linear attention, rewritten for float16: upstream sums ReLU(k) v over all
1,024 positions in float32. On the Neural Engine everything is float16, and
those sums pass 65,504. Keys are divided by the position count first; the
attention is a ratio of two such sums, so the scale cancels exactly.
"""
import sys, torch, numpy as np, coremltools as ct
import torch.nn.functional as F
from diffusers import SanaTransformer2DModel, AutoencoderDC

repo, out = sys.argv[1], sys.argv[2]
which = sys.argv[3] if len(sys.argv) > 3 else 'transformer'

# Upstream divides by (sum + 1e-15); where a head's ReLU'd keys are all
# zero that is 0 / 1e-15 = 0. In float16 the epsilon is below the smallest
# normal number, the GPU and Neural Engine flush it to zero, and 0 / 0 is
# NaN — the decoder's first stage came out NaN on the GPU. The smallest
# normal float16 keeps 0 / x = 0.
SMALLEST_NORMAL = 2.0 ** -14

def scaled_rms(x, norm, prescale=1.0):
    """`norm` (an RMSNorm over the last axis) of `x * prescale`, without
    float16 overflow: each vector is divided by its largest entry first and
    the epsilon by that squared, which leaves the result the same."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
    y = x / scale
    y = y * torch.rsqrt((y * y).mean(dim=-1, keepdim=True) + norm.eps / (prescale * scale) ** 2)
    return y * norm.weight if norm.weight is not None else y


# SANA-Sprint 1.6B's cross-attention queries reach 114,466 in its last blocks,
# past float16's 65,504, and its norms across heads squared them: the whole
# program came out NaN. The projection runs on input scaled by this, and the
# norm, which doesn't depend on scale, undoes it.
QUERY_PRESCALE = 8.0


def scaled_projection(linear, x, prescale):
    bias = linear.bias / prescale if linear.bias is not None else None
    return F.linear(x / prescale, linear.weight, bias)


class ScaledCrossAttention:
    """SANA's cross-attention (SanaAttnProcessor2_0) with its query
    projected at 1/QUERY_PRESCALE scale and overflow-free norms."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.norm_q is not None:
            query = scaled_rms(scaled_projection(attn.to_q, hidden_states, QUERY_PRESCALE), attn.norm_q, QUERY_PRESCALE)
        else:
            query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if attn.norm_k is not None:
            key = scaled_rms(key, attn.norm_k)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden = hidden.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](hidden))


class ScaledLinearAttention:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        if attn.norm_q is not None:
            query = scaled_rms(query, attn.norm_q)
        if attn.norm_k is not None:
            key = scaled_rms(key, attn.norm_k)
        n = hidden_states.shape[1]
        query = F.relu(query.transpose(1, 2).unflatten(1, (attn.heads, -1)))              # B H D N
        key = F.relu(key.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3)) / n  # B H N D
        value = value.transpose(1, 2).unflatten(1, (attn.heads, -1))                       # B H D N
        value = F.pad(value, (0, 0, 0, 1), mode='constant', value=1.0)                     # B H D+1 N
        scores = torch.matmul(value, key)                                                  # B H D+1 D
        hidden = torch.matmul(scores, query)                                               # B H D+1 N
        hidden = hidden[:, :, :-1] / hidden[:, :, -1:].clamp(min=SMALLEST_NORMAL)
        hidden = hidden.flatten(1, 2).transpose(1, 2)
        hidden = attn.to_out[0](hidden)
        return attn.to_out[1](hidden)

class ExpSiLU(torch.nn.Module):
    def forward(self, x):
        return x / (1 + torch.exp(-x))

def replace_silu(model):
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.SiLU):
                setattr(module, child_name, ExpSiLU())

class Transformer(torch.nn.Module):
    """The 28 blocks and the output projection. The timestep and guidance
    embedding (sin/cos and SiLU MLPs) and the caption projection (GELU and
    an RMSNorm) are computed outside in float32: on the Neural Engine their
    small errors reach every block, and the whole transformer agreed with
    float32 at 0.988; with them given as inputs, 0.9998."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, hidden, caption, caption_bias, modulation, embedded):
        m = self.model
        h = m.patch_embed(hidden)
        for block in m.transformer_blocks:
            h = block(h, None, caption, caption_bias, modulation, 32, 32)
        h = m.norm_out(h, embedded, m.scale_shift_table)
        h = m.proj_out(h)
        return h.reshape(1, 32, 32, 1, 1, -1).permute(0, 5, 1, 3, 2, 4).reshape(1, -1, 32, 32)

class ChannelNorm(torch.nn.Module):
    """DC-AE's RMSNorm over the channels of an NCHW tensor, in place of its
    transposes to channels-last and back, with each pixel divided by its
    largest channel first (see the top)."""
    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.bias = norm.bias
        self.eps = norm.eps
    def forward(self, x):
        scale = x.abs().amax(dim=1, keepdim=True).clamp(min=1.0)
        x = x / scale
        x = x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + self.eps / (scale * scale))
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

class DecoderTail(torch.nn.Module):
    """DC-AE's up blocks 2, 1 and 0 (128 to 1,024 px, ResBlocks only), its
    output norm and convolution: (1, 512, 128, 128) to (1, 3, 1024, 1024)."""
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.norms = torch.nn.ModuleDict()
        for i in (2, 1, 0):
            for j, block in enumerate(decoder.up_blocks[i][1:]):
                self.norms[f'{i}_{j}'] = ChannelNorm(block.norm)
        self.norms['out'] = ChannelNorm(decoder.norm_out)
    def forward(self, h):
        d = self.decoder
        for i in (2, 1, 0):
            up = d.up_blocks[i][0]
            # DCUpBlock2d: a convolution of the nearest-neighbour upsampling,
            # plus the input's channels repeated and pixel-shuffled.
            n, c, height, width = h.shape
            shortcut = h.unsqueeze(2).expand(n, c, up.repeats, height, width).reshape(n, c * up.repeats, height, width)
            h = up.conv(F.interpolate(h, scale_factor=2, mode='nearest')) + F.pixel_shuffle(shortcut, 2)
            for j, block in enumerate(d.up_blocks[i][1:]):
                h = self.norms[f'{i}_{j}'](block.conv2(block.nonlinearity(block.conv1(h)))) + h
        h = F.relu(self.norms['out'](h))
        return d.conv_out(h)

if which == 'transformer':
    model = SanaTransformer2DModel.from_pretrained(repo, subfolder='transformer', torch_dtype=torch.float32).eval()
    for block in model.transformer_blocks:
        block.attn1.set_processor(ScaledLinearAttention())
        if block.attn2 is not None and block.attn2.norm_q is not None:
            block.attn2.set_processor(ScaledCrossAttention())
    replace_silu(model)
    wrapper = Transformer(model).eval()
    width = model.config.num_attention_heads * model.config.attention_head_dim   # 1152 for 0.6B, 2240 for 1.6B
    example = (torch.randn(1, 32, 32, 32), torch.randn(1, 300, width), torch.zeros(1, 1, 300),
               torch.randn(1, 6 * width), torch.randn(1, width))
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    names = ['hidden', 'caption', 'caption_bias', 'modulation', 'embedded']
    mlmodel = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=e.shape, dtype=np.float32) for n, e in zip(names, example)],
                         outputs=[ct.TensorType(name='out', dtype=np.float32)],
                         compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE,
                         minimum_deployment_target=ct.target.iOS18, convert_to='mlprogram')
    mlmodel.save(f'{out}/sana_transformer.mlpackage')
else:
    vae = AutoencoderDC.from_pretrained(repo, subfolder='vae', torch_dtype=torch.float32).eval()
    replace_silu(vae)
    tail = DecoderTail(vae.decoder).eval()
    example = (torch.randn(1, 512, 128, 128),)
    with torch.no_grad():
        traced = torch.jit.trace(tail, example)
    mlmodel = ct.convert(traced, inputs=[ct.TensorType(name='hidden', shape=example[0].shape, dtype=np.float16)],
                         outputs=[ct.TensorType(name='image', dtype=np.float16)],
                         compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE,
                         minimum_deployment_target=ct.target.iOS18, convert_to='mlprogram')
    mlmodel.save(f'{out}/sana_decoder_tail.mlpackage')
print('saved', which)
