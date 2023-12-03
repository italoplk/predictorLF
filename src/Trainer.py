from argparse import Namespace

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from DataSet import DataSet, LensletBlockedReferencer


class Trainer:

    def __init__(self, dataset: DataSet, config_name: str, params: Namespace):
        self.model_name = params.model
        # TODO make loss GREAT AGAIN, nope, make it a param.
        self.loss = nn.MSELoss()
        self.params = params
        self.effective_predictor_size_v = self.params.num_views_ver * self.params.predictor_size
        self.effective_predictor_size_h = self.params.num_views_hor * self.params.predictor_size


        # TODO after everything else is done, adapt for other models
        self.model = ModelOracle(params.model).get_model(config_name, params)
        # TODO make AMERICA GREAT AGAIN, nope.... Num works be a parameter too
        # TODO test prefetch_factor and num_workers to optimize
        self.train_set = DataLoader(dataset.list_train, shuffle=True, num_workers=1,
                                    pin_memory=True, prefetch_factor=2)
        self.val_set = DataLoader(dataset.list_test, shuffle=False, num_workers=8,
                                  pin_memory=True)
        self.test_set = DataLoader(dataset.list_test, shuffle=False, num_workers=8,
                                   pin_memory=True)

        if torch.cuda.is_available():
            self.model.cuda()
            device = torch.device("cuda")
        else:
            print("Running on CPU!")
            device = torch.device("cpu")

        self.loss = self.loss.to(device)

        # TODO check betas
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr, betas=(0.9, 0.999))

        for epoch in range(1, 1 + params.epochs):
            loss = self.train(epoch, 0, params.wandb_active)
            print(f"Epoch {epoch}: {loss}")

            if params.wandb_active:
                wandb.log({f"Epoch": epoch})
                wandb.log({f"MSE_era": loss})

            loss = self.train(epoch, 1, params.wandb_active)
            print(f"Validation: {loss}")

            if params.wandb_active:
                wandb.log({f"MSE_VAL_era": loss})

            torch.save(self.model.state_dict(), f"/home/machado/saved_models/{config_name}.pth.tar")

    def train(self, current_epoch, val, wandb_active):
        acc = 0
        batches_now = 0
        if val == 0:
            self.model.train()
            set = self.train_set
        else:
            # TODO validation set
            set = self.test_set
            self.model.eval()

        for i, data in enumerate(set):
            # print(data.shape)
            # TODO ta fazendo 4 batches por lf apenas. Tamo fazendo soh 4 crop?
            # possible TODO: make MI_Size take a tuple
            referencer = LensletBlockedReferencer(data, data, MI_size=self.params.num_views_ver,
                                                  predictor_size=self.params.predictor_size)
            loader = DataLoader(referencer, batch_size=self.params.batch_size)

            for neighborhood, actual_block in loader:
                current_batch_size = actual_block.shape[0]
                if torch.cuda.is_available():
                    neighborhood, actual_block = (neighborhood.cuda(), actual_block.cuda())
                predicted = self.model(neighborhood)
                predicted = predicted[:, :, -self.effective_predictor_size_v:, -self.effective_predictor_size_h:]
                actual_block = actual_block[:, :, -self.effective_predictor_size_v:, -self.effective_predictor_size_h:]
                loss = self.loss(predicted, actual_block)
                if val == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                # loss = Mean over batches... so we weight the loss by the batch
                loss = loss.cpu().item()
                acc += loss * current_batch_size
                batches_now += current_batch_size
                if wandb_active:
                    if val == 0:
                        wandb.log({f"Batch_MSE_era_{current_epoch}": loss})
                        wandb.log({f"Batch_MSE_global": loss})
                    else:
                        wandb.log({f"Batch_MSE_VAL_global_{current_epoch}": loss})

        return acc / batches_now


class ModelOracle:
    def __init__(self, model_name):
        if model_name == 'Unet2k':
            from Models.latest_3k_5L_S2_1channel import UNetSpace
            # talvez faça mais sentido sò passar as variaveis necessarias do dataset
            self.model = UNetSpace
        if model_name == 'UnetGabriele':
            from Models.u3k_5L_S2_1view import UNetSpace
            # talvez faça mais sentido sò passar as variaveis necessarias do dataset
            print("3k")
            self.model = UNetSpace
        elif model_name == 'Unet3k':
            from Models.latest_3k_5L_S2_1channel import UNetSpace
            self.model = UNetSpace
        else:
            print("Model not Found.")
            exit(404)

    def get_model(self, config_name, params):
        try:
            return self.model(config_name, params)
        except RuntimeError as e:
            print("Failed to import model: ", e)
