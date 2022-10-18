docker run -it --rm --name tgbot \
--privileged=true \
--net=host \
stt_service:latest \
/bin/bash -c "/opt/conda/envs/stt_service/bin/run_tgbot"
