#!/bin/bash
set -e

echo "===== Python and pip versions ====="
python --version
pip --version

echo "===== Installed pip packages ====="
pip list

echo "===== Testing imports ====="
python -c "
try:
    print('Importing huggingface_hub...')
    import huggingface_hub
    print(f'huggingface_hub version: {huggingface_hub.__version__}')
    print('huggingface_hub imports available:', dir(huggingface_hub))
    
    print('\\nImporting diffusers...')
    import diffusers
    print(f'diffusers version: {diffusers.__version__}')
    
    print('\\nTesting specific imports...')
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    print('StableDiffusionPipeline and DPMSolverMultistepScheduler imported successfully')
    
    print('\\nAll imports successful!')
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    print(traceback.format_exc())
"

echo "===== GPU Info ====="
nvidia-smi

echo "===== Debug complete ====="