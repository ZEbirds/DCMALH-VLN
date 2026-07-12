docker run -it \
  --name grapheqa_habitat \
  --gpus all \
  -v "/home/csl-p920/Zibo Zheng/ZiboZHeng/graph_eqa:/workspace" \
  -v "/home/csl-p920/Zibo Zheng/ZiboZHeng/graph_eqa/grapheqa_data_backup:/root/graph_eqa/data" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY=$DISPLAY \
  blakerbuchanan/grapheqa_for_habitat:0.0.1 \
  bash

docker start tule
sudo docker exec -it tule bash
export PYTHONPATH=/workspace/graph_eqa:$PYTHONPATH
export PYTHONPATH="/workspace:$PYTHONPATH"
export OPENAI_BASE_URL=
export OPENAI_API_KEY=
cd workspace/