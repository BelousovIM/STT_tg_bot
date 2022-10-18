import json
import os
import tempfile

import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from tritonclient.grpc import service_pb2


def str_dtype2torch_dtype(model_dtype):
    if model_dtype == "BOOL":
        return bool
    elif model_dtype == "INT8":
        return np.int8
    elif model_dtype == "INT16":
        return np.int16
    elif model_dtype == "INT32":
        return np.int32
    elif model_dtype == "INT64":
        return np.int64
    elif model_dtype == "UINT8":
        return np.uint8
    elif model_dtype == "UINT16":
        return np.uint16
    elif model_dtype == "FP16":
        return np.float16
    elif model_dtype == "FP32":
        return np.float32
    elif model_dtype == "FP64":
        return np.float64
    elif model_dtype == "BYTES":
        return np.dtype(object)
    return None


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an STT network (as expected by
    this client)
    """
    assert (
        len(model_metadata.inputs) == 2
    ), f"expecting 1 input, got {len(model_metadata.inputs)}"
    assert (
        len(model_metadata.outputs) == 1
    ), f"expecting 1 output, got {len(model_metadata.outputs)}"
    assert (
        len(model_config.input) == 2
    ), f"expecting 1 input in model configuration, got {len(model_config.input)}"

    input_metadata = model_metadata.inputs[0]
    output_metadata = model_metadata.outputs[0]

    assert output_metadata.datatype == "FP32", (
        f"Expecting output datatype to be FP32, model {model_metadata.name!r} \n"
        f"Output type is {output_metadata.datatype}"
    )

    return (
        model_config.max_batch_size,
        (model_metadata.inputs[0].name, model_metadata.inputs[1].name),
        output_metadata.name,
        (model_config.input[0].format, model_config.input[1].format),
        (model_metadata.inputs[0].datatype, model_metadata.inputs[1].datatype),
    )


def preprocess(cfg, quartznet: nemo_asr.models.EncDecCTCModel):
    config = {
        "manifest_filepath": os.path.join(cfg["temp_dir"], "manifest.json"),
        "sample_rate": 16000,
        "labels": quartznet.decoder.vocabulary,
        "batch_size": min(cfg["batch_size"], len(cfg["paths2audio_files"])),
        "trim_silence": True,
        "shuffle": False,
    }
    dataset = AudioToCharDataset(
        manifest_filepath=config["manifest_filepath"],
        labels=config["labels"],
        sample_rate=config["sample_rate"],
        int_values=config.get("int_values", False),
        augmentor=None,
        max_duration=config.get("max_duration", None),
        min_duration=config.get("min_duration", None),
        max_utts=config.get("max_utts", 0),
        blank_index=config.get("blank_index", -1),
        unk_index=config.get("unk_index", -1),
        normalize=config.get("normalize_transcripts", False),
        trim=config.get("trim_silence", True),
        parser=config.get("parser", "en"),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        collate_fn=dataset.collate_fn,
        drop_last=config.get("drop_last", False),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )
    return dataloader


def requestGenerator(
    input_names, output_name, datatypes, args, quartznet,
):
    request = service_pb2.ModelInferRequest()
    request.model_name = args["model_name"]
    request.model_version = args["model_version"]

    filenames = [
        args["audio_filename"],
    ]

    output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output.name = output_name
    request.outputs.extend([output])

    input_0 = service_pb2.ModelInferRequest().InferInputTensor()
    input_0.name = input_names[0]
    input_0.datatype = datatypes[0]

    input_1 = service_pb2.ModelInferRequest().InferInputTensor()
    input_1.name = input_names[1]
    input_1.datatype = datatypes[1]

    to_numpy = (
        lambda tensor: tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "manifest.json"), "w") as fp:
            for audio_filepath in filenames:
                entry = {
                    "audio_filepath": audio_filepath,
                    "duration": 100000,
                    "text": "nothing",
                }
                fp.write(json.dumps(entry) + "\n")

        config = {
            "paths2audio_files": filenames,
            "batch_size": args["batch_size"],
            "temp_dir": tmpdir,
        }
        for test_batch in preprocess(config, quartznet):
            input_0.shape.extend(
                [
                    args["batch_size"],
                    test_batch[1].to(quartznet.device).max().item(),
                ]
            )
            input_1.shape.extend([args["batch_size"], 1])
            raw_input_contents = [
                to_numpy(test_batch[0].to(quartznet.device)).tobytes(),
                to_numpy(test_batch[1].to(quartznet.device)).tobytes(),
            ]
            request.inputs.extend([input_0, input_1])
            request.raw_input_contents.extend(raw_input_contents)
            yield request


def postprocess(response, quartznet) -> str:
    """
    Post-process response to show classifications.
    """
    if len(response.outputs) != 1:
        raise Exception(f"expected 1 output, got {len(response.outputs)}")

    if len(response.raw_output_contents) != 1:
        raise Exception(
            f"expected 1 output content, got {len(response.raw_output_contents)}"
        )

    buffer = response.raw_output_contents[0]
    dtype = str_dtype2torch_dtype(response.outputs[0].datatype)
    shape = tuple(response.outputs[0].shape)
    alogits = np.frombuffer(buffer, dtype=dtype).reshape(shape)
    logits = torch.tensor(alogits)

    greedy_predictions = logits.argmax(dim=-1, keepdim=False)
    hypotheses, _ = quartznet.decoding.ctc_decoder_predictions_tensor(
        greedy_predictions
    )
    return hypotheses[0]
