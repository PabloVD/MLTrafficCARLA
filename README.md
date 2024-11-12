# MLTrafficCARLA

**WORK IN PROGRESS**

Data-driven traffic generator based on the deep learning models from [this repo](https://github.com/PabloVD/TrafficTrainer), trained in the [Waymo Open Motion Dataset](https://waymo.com/open/).

## Usage

Start a CARLA simulation and run `python3 run_nn_traffic.py -m PATH_OF_MODEL -npcs NUM_OF_NPCS`.

![Example of rasterized road, used as input by the neural network](road.png)