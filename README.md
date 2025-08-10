# How to patch Or Download Custom Docker
**Patch**
1. Edit $CONTAINER_NAME in frigate_build_mod.sh
2. Run original frigate docker #Same $CONTAINER_NAME
3. Run frigate_build_mod.sh

**Download**

docker pull cluangar/frigate:0.16.0-rc2-npu

# Example Run Docker
sudo docker run -it --rm --privileged \
--security-opt systempaths=unconfined \
--security-opt apparmor=unconfined \
--device /dev/dri:/dev/dri \
--device /dev/dma_heap:/dev/dma_heap \
--device /dev/accel:/dev/accel \
-v $(pwd)/config:/config \
-v $(pwd)/media:/media/frigate \
-v /etc/localtime:/etc/localtime:ro \
-p 5000:5000 \
-p 8971:8971 \
-p 8554:8554 \
-p 8555:8555/tcp \
-p 8555:8555/udp \
-p 1984:1984/tcp \
--shm-size=300m  \
--tmpfs /tmp/cache:size=1G \
--cap-add=CAP_PERFMON \
frigate:0.16.0-rc2-npu

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6c334aba-91f2-4e22-88ac-b068e5b5c637" />

