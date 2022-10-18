docker run --rm -p 5000:8000 -p 5001:8001 -p 5002:8002 \
-v "${PWD}"/model_repository:/models nvcr.io/nvidia/tritonserver:21.09-py3 \
tritonserver --model-repository=/models