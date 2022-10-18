# STT Tg Service
tg bot: https://t.me/mynalabs_stt_service_bot

## Installation 

### *Install repository
```shell
git clone https://gitlab.com/BelousovIM/mynalabs.git
cd mynalabs && conda env create -f conda-environment.yaml 
conda activate stt_service
```
*you don't have to install repository to run current service

### Build container
This step is necessary to run STT client and tg bot api.

```shell
docker build -t stt_service .
```

## Run containers
### Run triton with QuartzNet15x5
```shell
sh sh/run_triton_service.sh
```
### Run STT client
```shell
sh sh/run_stt_client.sh
```
And run inside the container:
```shell
run_stt_client
```
### Run telegram bot frontend
```shell
sh sh/run_tgbot.sh
```