PORT=30001
GPU=$1
CFG=$2
TAG=${3:-'default'}

torchrun --nproc_per_node $GPU --master_port $PORT /kaggle/input/dat-training/main.py --cfg $CFG --data-path '/kaggle/input/imagenetmini-1000/imagenet-mini' --amp --tag $TAG