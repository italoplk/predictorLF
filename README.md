**LEARNING BASED LIGHT FIELD INTRA PREDICTOR TRAINING**


**To start training run the Runner.py file.**

**Project Structure**
  Runner.py: Responsible for starting the trainings, directories and names.
  Dataset.py: Responsible for loading the data and defining training and validation cases. Also defines how the LFs will be patched.
  Lightfield.py: Responsible for image operations as: normalize, load, save, etc.
  Trainer: Responsible for training, pruning, loading models, saving models, saving blocks.
  Params: Has all the cmd params, useful to learn how to properly configure the trainings.

  Files starting with lower case are responsible for small functions as customized losses, quantizations, prunning, etc.

**Important parameters:**
  --std-path: Root directory where outputs should be saved. The folders "saved_models/{pruning_opt}/{run-name}" will be created. Also "saved_LFs" in case you choose to --save-train or to --save-validation. 
  (all models will be saved with names based on their configurations)
  --dataset-path: Direcory with training Lenslet gray scale Light Fields
  --run-name: An text specifier that will be attached to the start of the config name. 
  --context-size: 'Size of the context [8, 16,32,64] (default 64)
  --predictor-size: Size of the predictor [8,16, 32] (default 32)
  --epochs: Number of epochs (default 100)

  --model: CNN arquitecture to be used in training. (Check Trainer.py for all possible options)
  --num-filters:  #filters to be used in the CNN layers. (Are multiplied by up to *8 in the last layers)
  --skip-connections: To apply skip connections in the architectures. (appliable to most architectures) (Skip, residual_connections, default= noSkip) 

**To resume a training:**
  --resume-epoch: To restart training a model from a given epoch
  --resume: Model (.tar) to have its training resumed
**
**WARNING:****
  The validation file names are hardcoded at the Dataset.py. While back a function "random_split" was coded to randomly pick the validation cases. Since we don't use such approach, it may be currently buggy.

**For quick tests:**
  --limit-train: Sets a limited number of light fields to train. (-1 to use all available)
  --limit-val: Sets a limited number of light fields to validate. (-1 to use all available)

**To prune:**
    --prune: If the model should be pruned. (After #epochs it is pruned, the LR rewinded and trained for another #epochs) 
    --target-sparsity: Final model sparsity
    --prune-step: How many parameters to prune each step
