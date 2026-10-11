"""DreamShaper 8 with Hyper-SD's 4-step LoRA, to Core ML for the Neural Engine.

    python convert_sd.py <work dir> <stage dir>

1. Fuses ByteDance's Hyper-SD15-4steps LoRA (CreativeML OpenRAIL++-M) into
   Lykon's DreamShaper 8 (CreativeML OpenRAIL-M), saved as float32.
2. Converts text encoder, UNet and VAE decoder with Apple's ml-stable-diffusion
   converter (`python_coreml_stable_diffusion.torch2coreml`; clone
   github.com/apple/ml-stable-diffusion and install it): SPLIT_EINSUM_V2
   attention for the Neural Engine, and a batch-one UNet — Hyper-SD needs no
   guidance, so there is no second, unconditioned image to batch.
3. Palettizes the text encoder and UNet to 8 bits, one table per tensor
   (kmeans). 6 bits drew a different picture from the same noise (a shirt
   where float16 drew a suit); 8 bits match float16 as closely as float16
   matches PyTorch. The VAE decoder stays float16.
4. Hands the three to finish_coreml.py: clean metadata, compile, archive.

The app adds CLIP's vocabulary and merges (clip_vocab.json, clip_merges.txt,
from the model's tokenizer folder) and samples with DDIM's "trailing" steps
999, 749, 499, 249 (SDPipeline.swift).

Converter quirks met with ml-stable-diffusion 1.1.0 and diffusers 0.41:
its optional `diffusionkit` import must be made optional; it asks for the
fp16 variant, so the fused model is saved with `variant='fp16'` too; it
passes `use_auth_token`, which newer diffusers rejects; and numpy above 2.2
breaks coremltools 9's cast ("only 0-dimensional arrays").
"""
import glob, os, subprocess, sys
import torch
import coremltools as ct
from coremltools.optimize.coreml import OptimizationConfig, OpPalettizerConfig, palettize_weights
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download

work, stage = sys.argv[1], sys.argv[2]
fused = os.path.join(work, 'dreamshaper8-hyper4')
converted = os.path.join(work, 'coreml-f16')
compressed = os.path.join(work, 'coreml-q8')
for d in (work, converted, compressed):
    os.makedirs(d, exist_ok=True)

# 1. Fuse ---------------------------------------------------------------------
lora = hf_hub_download('ByteDance/Hyper-SD', 'Hyper-SD15-4steps-lora.safetensors')
pipe = StableDiffusionPipeline.from_pretrained('Lykon/dreamshaper-8', torch_dtype=torch.float16, variant='fp16',
                                               safety_checker=None, requires_safety_checker=False)
pipe.load_lora_weights(lora, adapter_name='hyper')
pipe.fuse_lora()
pipe.unload_lora_weights()
pipe.to(torch.float32)
pipe.save_pretrained(fused, safe_serialization=True)
pipe.save_pretrained(fused, safe_serialization=True, variant='fp16')

# 2. Convert -------------------------------------------------------------------
subprocess.run([sys.executable, '-m', 'python_coreml_stable_diffusion.torch2coreml',
                '--model-version', fused, '--convert-unet', '--convert-text-encoder', '--convert-vae-decoder',
                '--attention-implementation', 'SPLIT_EINSUM_V2', '--compute-unit', 'CPU_AND_NE',
                '--unet-batch-one', '-o', converted], check=True)


def package(part):
    return glob.glob(os.path.join(converted, f'*_{part}.mlpackage'))[0]


# 3. Palettize -------------------------------------------------------------------
config = OptimizationConfig(global_config=OpPalettizerConfig(mode='kmeans', nbits=8, weight_threshold=2048))
outputs = {}
for part in ('text_encoder', 'unet'):
    model = ct.models.MLModel(package(part), skip_model_load=True)
    outputs[part] = os.path.join(compressed, f'{part}.mlpackage')
    palettize_weights(model, config).save(outputs[part])
outputs['vae_decoder'] = package('vae_decoder')

# 4. Finish ------------------------------------------------------------------------
licence = 'DreamShaper 8 (Lykon) with Hyper-SD (ByteDance): CreativeML OpenRAIL-M / OpenRAIL++-M'
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'finish_coreml.py'), stage,
                f'sd_text_encoder={outputs["text_encoder"]}:{licence}',
                f'sd_unet={outputs["unet"]}:{licence}',
                f'sd_vae_decoder={outputs["vae_decoder"]}:{licence}'], check=True)
