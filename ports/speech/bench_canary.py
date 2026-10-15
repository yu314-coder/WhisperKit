# Canary 1B v2 (8-bit MLX, GPU) on the English FLEURS clips; it has no Chinese or Japanese.
import glob, json, os, time
from mlx_audio.stt.utils import load_model
model = load_model('Mediform/canary-1b-v2-mlx-q8')
model.generate(sorted(glob.glob('en_us/*.wav'))[0], source_lang='en', target_lang='en')
os.makedirs('out/canary-1b-v2', exist_ok=True)
texts, start = {}, time.time()
for wav in sorted(glob.glob('en_us/*.wav')):
    texts[os.path.basename(wav)] = model.generate(wav, source_lang='en', target_lang='en').text
json.dump(texts, open('out/canary-1b-v2/en_us.json', 'w'), ensure_ascii=False, indent=0)
print('canary-1b-v2', 'en_us', time.time() - start, flush=True)
