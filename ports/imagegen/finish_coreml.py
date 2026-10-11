"""Gives converted Core ML packages clean metadata, compiles them and packs
each compiled model into an Apple Archive for GitHub Releases.

    python finish_coreml.py <stage dir> <name>=<path.mlpackage>[:<licence>] ...

Apple's converter records the local checkpoint path as the model's version
and author; this replaces them with what the model is and where it is from.
"""
import os, subprocess, sys, shutil, tempfile
import coremltools as ct

stage = sys.argv[1]
os.makedirs(stage, exist_ok=True)
for spec in sys.argv[2:]:
    name, rest = spec.split('=', 1)
    path, _, licence = rest.partition(':')
    model = ct.models.MLModel(path, skip_model_load=True)
    model.author = 'Converted for WhisperKit (github.com/yu314-coder/WhisperKit)'
    model.version = name
    model.short_description = name
    if licence:
        model.license = licence
    with tempfile.TemporaryDirectory() as tmp:
        package = os.path.join(tmp, name + '.mlpackage')
        model.save(package)
        compiled = os.path.join(stage, name + '.mlmodelc')
        shutil.rmtree(compiled, ignore_errors=True)
        subprocess.run(['xcrun', 'coremlcompiler', 'compile', package, tmp], check=True, capture_output=True)
        shutil.move(os.path.join(tmp, name + '.mlmodelc'), compiled)
    archive = compiled + '.aar'
    if os.path.exists(archive):
        os.remove(archive)
    subprocess.run(['aa', 'archive', '-d', stage, '-subdir', name + '.mlmodelc', '-o', archive, '-a', 'lzfse'], check=True)
    print(name, os.path.getsize(archive))
