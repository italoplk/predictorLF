#FROM eidos-service.di.unito.it/eidos-base-pytorch:1.10.0
#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM eidos-service.di.unito.it/eidos-base-pytorch:1.12.0
#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Create the environment:
RUN apt update -y
RUN apt install -y gcc
RUN apt install -y g++ 

RUN pip install compressai 
RUN pip install wandb
RUN pip install torchac
#RUN pip install ipywidgets
#RUN pip install Ninja
RUN pip install pytest-gc
#RUN pip install timm
#RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric torchvision -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install albumentations
RUN pip install Pillow
RUN pip install einops
RUN pip install torchsummary
RUN pip install pytorch_ssim

#RUN addgroup --gid 1337 eidoslab

# This is wandb stuff
RUN mkdir -p /.config
RUN chmod 775 /.config
RUN chown -R :1337 /.config

#For pytorch checkpoints
RUN mkdir -p /.cache
RUN chmod 775 /.cache
RUN chown -R :1337 /.cache

# Copy source files and make it owned by the group eidoslab
# and give write permission to the group
COPY ./ /src
RUN chmod 775 /src
RUN chown -R :1337 /src

WORKDIR /src

ENTRYPOINT ["python3"]