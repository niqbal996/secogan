## Content Disentanglement for Semantically Consistent Synthetic-to-Real Domain Adaptation

Code for the paper:

Mert Keser*, Artem Savkin*, [Federico Tombari](http://campar.in.tum.de/Main/FedericoTombari), "[Content Disentanglement for Semantically Consistent Synthetic-to-Real Domain Adaptation](https://arxiv.org/abs/2105.08704)", IEEE IROS 2021

### Requirements

Dockerfile:

```
FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04
RUN apt-get update
RUN apt-get update && apt-get install -y python3-dev python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.3.1
RUN pip3 install torchvision==0.4.2
```

### Usage

Train command:

```
python secogan/train.py \
    --name=<experiment_name> \
    --gpu_ids=0 \
    --data_source=<source_data_path> \
    --data_target=<target_data_path> \
    --output_dir=<output_path> \
    --batch_size=4
```

### Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{Keser2021,
    title={Content Disentanglement for Semantically Consistent Synthetic-to-Real Domain Adaptation},
    author={Keser, Mert and Savkin, Artem and Tombari, Federico},
    booktitle={IEEE IROS},
    year={2021}
}
```

### Acknowledgments
This repo heavily borrows code from [MUNIT](https://github.com/NVlabs/MUNIT), [SAE](https://github.com/taesungp/swapping-autoencoder-pytorch).