
export N_NPCS=10

export MODELPATH="/home/tda/CARLA/TrafficGeneration/TrafficTrainer/models/"

#export MODEL=NLL_20_vit_large_patch32_224_closedloop_0
export MODEL=NLL_20_vit_large_patch32_224_cumsum_2

python3 run_nn_traffic.py \
	-m $MODELPATH$MODEL \
	-npcs $N_NPCS

