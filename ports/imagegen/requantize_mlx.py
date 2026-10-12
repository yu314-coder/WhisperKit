"""Re-quantizes 8-bit MLX weight parts to fewer bits, keeping everything else.

    python requantize_mlx.py <in stem> <out stem> <bits> [notice]

Reads `<in stem>.partN.safetensors` (from the export scripts here: each
quantized matrix as `.wq`/`.scales`/`.biases`, groups of 64), and writes
`<out stem>.partN.safetensors` with every matrix at `bits` — scales and
biases in the type they had — 64-byte aligned for MappedWeights.swift.
`notice`, when given, goes in each file's header (the Qwen licence asks
changed files to say so).

Going 8 → 4 bits through the 8-bit weights rather than the originals adds
the 8-bit rounding (a 256th of each group's range) to the 4-bit rounding
(a 16th): next to nothing.
"""
import glob, os, sys
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safetensors_aligned import save_parts  # noqa: E402

source, target, bits = sys.argv[1], sys.argv[2], int(sys.argv[3])
notice = sys.argv[4] if len(sys.argv) > 4 else None

tensors = {}
for path in sorted(glob.glob(f'{source}.part*.safetensors')):
    tensors.update(mx.load(path))
result = {}
for name in sorted(n for n in tensors if n.endswith('.wq')):
    base = name[:-3]
    scales, biases = tensors.pop(base + '.scales'), tensors.pop(base + '.biases')
    full = mx.dequantize(tensors.pop(name), scales.astype(mx.float32), biases.astype(mx.float32),
                         group_size=64, bits=8)
    q, s, b = mx.quantize(full, group_size=64, bits=bits)
    result[base + '.wq'], result[base + '.scales'], result[base + '.biases'] = q, s.astype(scales.dtype), b.astype(biases.dtype)
    mx.eval(result[base + '.wq'], result[base + '.scales'], result[base + '.biases'])
result.update(tensors)
print('parts:', save_parts(target, result, {'notice': notice} if notice else None))
