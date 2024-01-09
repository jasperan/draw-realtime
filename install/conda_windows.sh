# you will need to manually install node.js
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
pip install --force-reinstall pywin32

