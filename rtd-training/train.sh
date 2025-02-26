echo "Training is alive!"

git clone https://github.com/saadam1n/rtdenoise

mv rtdenoise/* *

pip install . -v

python3 -u scripts/sample_train.py