"""The Image tab's filter for sexually explicit pictures, to Core ML.

    python convert_safety.py <Falconsai/nsfw_image_detection dir> <out.mlpackage>

Falconsai's ViT-base classifier (Apache 2.0): one picture, 224 × 224, to
the probabilities of "normal" and "nsfw". The input is RGB in 0...1,
channels first; the model's own normalization ((x - 0.5) / 0.5) is inside.
It runs on the Neural Engine in float16.

This is the only check on what the Image tab makes: prompts are never
filtered, so politics, public figures and satire all work. App Store
guideline 1.1.4 rules out overtly sexual material, and this keeps such a
picture from being shown or saved.
"""
import sys
import numpy as np
import torch
import coremltools as ct
from transformers import ViTForImageClassification

source, out = sys.argv[1], sys.argv[2]
model = ViTForImageClassification.from_pretrained(source, torch_dtype=torch.float32).eval()


class Classifier(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixels):
        return torch.softmax(self.model((pixels - 0.5) / 0.5).logits, dim=-1)


example = torch.rand(1, 3, 224, 224)
with torch.no_grad():
    traced = torch.jit.trace(Classifier(model).eval(), example)
mlmodel = ct.convert(traced, inputs=[ct.TensorType(name='pixels', shape=example.shape, dtype=np.float32)],
                     outputs=[ct.TensorType(name='probabilities', dtype=np.float32)],
                     compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE,
                     minimum_deployment_target=ct.target.iOS18, convert_to='mlprogram')
mlmodel.save(out)
print('saved', out)
