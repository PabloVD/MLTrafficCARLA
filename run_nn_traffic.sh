
export N_NPCS=9

export MODELPATH="/home/tda/CARLA/TrafficGeneration/TrafficTrainer/models/"
export MODEL=L1_80_egotransf_0

python3 run_nn_traffic.py \
	-m $MODELPATH$MODEL \
	-npcs $N_NPCS

