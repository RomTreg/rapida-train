FROM continuumio/miniconda3


RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y git
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install nano  -y
RUN mkdir -p /home/ubuntu/segm_code/

WORKDIR /home/ubuntu/segm_code/


COPY ./spec-file.txt /home/ubuntu/segm_code
COPY ./pip_requirements.txt /home/ubuntu/segm_code
COPY ./train.py /home/ubuntu/segm_code
COPY ./waiter.py /home/ubuntu/segm_code
COPY ./maskOutput.py /home/ubuntu/segm_code
COPY ./prepDataSet.py /home/ubuntu/segm_code
RUN conda create --name tf-seg --file spec-file.txt

RUN echo "conda activate tf-seg" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN conda activate tf-seg
RUN rm -rf ~/.nv/
RUN conda install cudatoolkit -y
RUN pip install -r pip_requirements.txt
RUN pip install --upgrade git+https://github.com/divamgupta/image-segmentation-keras

ENTRYPOINT ["conda", "run", "-n", "tf-seg", "python", "waiter.py"]