docker run -it --rm --name stt_client \
--privileged=true \
--net=host \
-v /Users:/Users \
stt_service:latest \
/bin/bash