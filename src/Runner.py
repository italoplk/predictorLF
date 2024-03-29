
import wandb

from DataSet import DataSet
from Params import get_args
from Trainer import Trainer

import os

def main():
    params = get_args()

    if params.run_name != 'test_dump':  
        config_name = f"{params.run_name}_{params.model}_{params.loss}_predS{params.predictor_size}_{params.batch_size}_{params.lr}"
    else:
        config_name = 'test_dump'
   
    try:
    
        os.mkdir(f"{params.std_path}/saved_LFs/{config_name}")
        os.mkdir(f"{params.std_path}/saved_LFs/{config_name}/validation/")
        os.mkdir(f"{params.std_path}/saved_models/{config_name}")

            
        if params.save_train:
            os.mkdir(f"{params.std_path}/saved_LFs/{config_name}/train/")
    except FileExistsError:
        print("Using Existent folder!!")




    dataset = DataSet(params)
    dataset.split()

    
    if params.wandb_active:

        wandb.login(key="9a53bad34073a4b6bcfa6c2cb67a857e976d86c4" ,force=True)

        wandb.init(
            # set the wandb project where this run will be logged
            project="predictorUnet",
            # track hyperparameters and run metadata
            name=config_name,
            config={
                "architecture": params.model,
                "dataset": params.dataset_name,
                "views ver": params.num_views_ver,
                "views hor": params.num_views_hor,
                "epochs": params.epochs,
                "batch size": params.batch_size,
                "learning_rate": params.lr,
                "Loss": params.loss,
                "scheduler": params.lr_scheduler,
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
                "Crop-mode": params.crop_mode


            }
        )



    # for lf in dataset.list_test.inner_storage:
    #     print(lf.name)


    Trainer(dataset, config_name, params)

    if params.wandb_active:
        wandb.finish()








if __name__ == '__main__':
    main()



