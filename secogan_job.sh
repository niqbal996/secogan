srun -K --ntasks=1 --gpus-per-task=2 -N 1 --cpus-per-gpu=20 -p A100-IML --mem=40000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/secogan:/home/iqbal/secogan \
  --container-image=/netscratch/naeem/nvcr.io_nvidia_pytorch_23.01-py3_secogan.sqsh \
  --container-workdir=/home/iqbal/secogan \
  --time=3-00:00 \
  bash train_secogan.sh
  # --pty /bin/bash
   