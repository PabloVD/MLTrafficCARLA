export MODEL=nll_80
export MODEL=L2_80_egotransf_0
export MODEL=L1_80_1
export MODEL=L2_80_0
export MODEL=NLL_80_egotransf_0
#export MODEL=NLL_80_otherstransf_1

python3 test_model.py \
	-m $MODEL \
	--data /home/tda/CARLA/TrafficGeneration/Datasets/Waymo_tf_example/rendered_valid_fixed	\
	--save ./viz

