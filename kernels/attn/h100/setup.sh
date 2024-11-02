pip install tqdm tabulate git+https://github.com/fkodom/grouped-query-attention-pytorch.git
python gentests.py 3072 64 24 24
sed -i 's/cuda-12.6/cuda-12.4/g' Makefile

