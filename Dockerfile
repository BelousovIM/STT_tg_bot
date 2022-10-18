FROM continuumio/miniconda3:4.10.3

COPY . /mynalabs
WORKDIR /mynalabs

# Non-internal gitlab dependencies and conda environment
RUN conda config --set ssl_verify no && conda env create --file conda-environment.yaml && \
    echo ". /opt/conda/etc/profile.d/conda.sh\nconda activate stt_service\n" > ~/.bashrc

SHELL ["/bin/bash", "-c", "--login"]

# Default entrypoint
CMD ["/bin/bash", "--login"]
