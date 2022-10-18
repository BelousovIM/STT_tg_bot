import yaml

import grpc
from omegaconf import DictConfig
import nemo.collections.asr as nemo_asr
from tritonclient.grpc import service_pb2, service_pb2_grpc

from stt import data_path
from stt.utils import parse_model, postprocess, requestGenerator


def stt_client(
    audio_filename: str,
    model_name: str,
    model_version: str = "",
    batch_size: int = 1,
    url: str = "localhost:5001",
) -> str:
    with (data_path / "configs/config.yaml").open() as file:
        params = yaml.safe_load(file)
    quartznet = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params["model"]))

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    metadata_request = service_pb2.ModelMetadataRequest(
        name=model_name, version=model_version
    )
    metadata_response = grpc_stub.ModelMetadata(metadata_request)

    config_request = service_pb2.ModelConfigRequest(
        name=model_name, version=model_version
    )
    config_response = grpc_stub.ModelConfig(config_request)

    (
        max_batch_size,
        input_names,
        output_name,
        input_formats,
        datatypes,
    ) = parse_model(
        model_metadata=metadata_response, model_config=config_response.config
    )

    if not (max_batch_size == 0) and batch_size != 1:
        raise Exception("This model doesn't support batching.")

    # Send request
    args = {
        "model_name": model_name,
        "model_version": model_version,
        "audio_filename": audio_filename,
        "batch_size": batch_size,
    }
    requests = requestGenerator(
        input_names=input_names,
        output_name=output_name,
        datatypes=datatypes,
        args=args,
        quartznet=quartznet,
    )

    responses = [grpc_stub.ModelInfer(request) for request in requests]

    response_text = None
    for response in responses:
        response_text = postprocess(response=response, quartznet=quartznet,)

    return response_text
