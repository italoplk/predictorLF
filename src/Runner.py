
import wandb

from DataSet import DataSet
from Params import get_args
from Trainer import Trainer

import os

def main():
    params = get_args()

    os.makedirs(f"{params.std_path}/saved_models/", exist_ok=True)
    os.makedirs(f"{params.std_path}/saved_models/pruned", exist_ok=True)
    os.makedirs(f"{params.std_path}/saved_models/unpruned", exist_ok=True)
        
        

    if params.run_name != 'test_dump':

        if params.prune:
            params.run_name = f"{params.run_name}_pruneS-T_{params.prune_step}-{params.target_sparsity}"
        config_name = f"{params.run_name}_{params.model}_{params.skip_connections}_{params.loss}_predS{params.predictor_size}_{params.batch_size}_{params.lr}_G{params.lr_gamma}"
        print(config_name)
    else:
        config_name = 'test_dump'
   
    try:
        if params.prune:
            os.mkdir(f"{params.std_path}/saved_models/pruned/{config_name}")
        else:
            os.mkdir(f"{params.std_path}/saved_models/unpruned/{config_name}")

            
        if params.save_test:
            os.mkdir(f"{params.std_path}/saved_LFs/{config_name}")
            os.mkdir(f"{params.std_path}/saved_LFs/{config_name}/validation/")
        if params.save_train:
            os.mkdir(f"{params.std_path}/saved_LFs/{config_name}")
            os.mkdir(f"{params.std_path}/saved_LFs/{config_name}/train/")

    except FileExistsError:
        print("Using Existent folder!!")




    dataset = DataSet(params)
    dataset.split()

    
    if params.wandb_active:

        wandb.login(key="9a53bad34073a4b6bcfa6c2cb67a857e976d86c4" ,force=True)

        wandb.init(
           
            project="predictorUnet",
            name=config_name,
            config={
                "architecture": params.model,
                "dataset": params.dataset_name,
                "dataset name": params.dataset_path,
                "views ver": params.num_views_ver,
                "views hor": params.num_views_hor,
                "epochs": params.epochs,
                "batch size": params.batch_size,
                "learning_rate": params.lr,
                "Loss": params.loss,
                "scheduler": params.lr_scheduler,
                "lr-gamma": params.lr_gamma,
                "lr-step": params.lr_step_size,
                "optimizer": params.optimizer,
                "name": config_name,
                "Training Size": len(dataset.list_train),
                "Test Size": len(dataset.list_test),
                "Bit-Depth": dataset.bit_depth,
                "Model": params.model,
                "Skip Connections": params.skip_connections,
                "Num-Filters": params.num_filters,
                "Context Size": params.context_size,
                "Predictor Size": params.predictor_size,
                "Transforms": params.transforms,
                "Crop-mode": params.crop_mode,
                "Prune Step": params.prune_step,
                "Target Sparsity": params.target_sparsity


            }
        )

    Trainer(dataset, config_name, params)

    if params.wandb_active:
        wandb.finish()


if __name__ == '__main__':
    main()



