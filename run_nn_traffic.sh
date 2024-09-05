
export N_NPCS=5

export MODELPATH="/home/tda/CARLA/TrafficGeneration/TrafficTrainer/models/"


export MODEL=closedloop_NLL_20_vit_base_patch16_224
#export MODEL=moredata_NLL_20_vit_base_patch16_224_1

python3 run_nn_traffic.py \
	-m $MODELPATH$MODEL \
	-npcs $N_NPCS

