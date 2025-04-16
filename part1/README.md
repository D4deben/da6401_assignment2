# Deep Learning Assignment 2 - CNN on iNaturalist Dataset

## How to Run
#-------------------------------#
### Training                    #
#-------------------------------#
To run this with default parameters:
# Default configuration (best parameters):
python train.py \
    --wandb_entity your_wandb_username \
    --wandb_project your_project_name \
    --data_dir path/to/inaturalist_12K/train

# Custom configuration example:
python train.py \
    --wandb_entity your_username \
    --wandb_project cnn-experiments \
    --epochs 25 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --optimizer adam \
    --weight_decay 1e-4

    
#-----------------#   
### Evaluation    #
#-----------------#
python evaluate.py \
    --model_path saved_models/best_model.pth \
    --test_dir path/to/test_data \
    --wandb_entity <your-wandb-username>
 
 
#-------------------#   
### Visualization   #
#-------------------#

python visualize.py \
    --model_path saved_models/best_model.pth \
    --test_dir path/to/test_data \
    --wandb_entity <your-wandb-username> 
    
    
#-------------------------#
###Best Hyperparameters   #
#-------------------------#
Parameter	Value
Batch Size	256
Learning Rate	0.01
Optimizer	SGD
Momentum	0.9
Weight Decay	1e-5
Epochs	20
Dropout	0.5
