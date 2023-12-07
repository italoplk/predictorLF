from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import wandb
from PIL import Image
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
        self.dataset = dataset
        self.best_loss = 1000.1


        # TODO REMOVE
        self.count_blocks = 0

        # TODO after everything else is done, adapt for other models
        self.model = ModelOracle(params.model).get_model(config_name, params)
        # TODO make AMERICA GREAT AGAIN, nope.... Num works be a parameter too
        # TODO test prefetch_factor and num_workers to optimize
        self.train_set = DataLoader(dataset.list_train, shuffle=True, num_workers=8,
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
                wandb.log({f"MSE_epoch": loss})

            # if epoch == 5:
            loss = self.train(epoch, 1, params.wandb_active)
            print(f"Validation: {loss}")

            if params.wandb_active:
                wandb.log({f"MSE_VAL_epoch": loss})

            if loss < self.best_loss:
                torch.save(self.model.state_dict(), f"/home/machado/saved_models/bestMSE_{config_name}_{epoch}.pth.tar")

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
            resol_ver = self.params.resol_ver
            resol_hor = self.params.resol_hor
            output_lf = torch.zeros((1, resol_ver, resol_hor))
            it_i = 0
            it_j = 0
            self.count_blocks = 0

        for i, data in enumerate(set):
            # print(data.shape)
            # TODO ta fazendo 4 batches por lf apenas. Tamo fazendo soh 4 crop?
            # possible TODO: make MI_Size take a tuple
            # print("data:", data.shape)
            referencer = LensletBlockedReferencer(data, MI_size=self.params.num_views_ver,
                                                  predictor_size=self.params.predictor_size)
            loader = DataLoader(referencer, batch_size=self.params.batch_size)

            for neighborhood, actual_block in loader:
                current_batch_size = actual_block.shape[0]
                if torch.cuda.is_available():
                    neighborhood, actual_block = (neighborhood.cuda(), actual_block.cuda())
                predicted = self.model(neighborhood)
                predicted = predicted[:, :, -self.effective_predictor_size_v:, -self.effective_predictor_size_h:]
                actual_block = actual_block[:, :, -self.effective_predictor_size_v:, -self.effective_predictor_size_h:]

                if val == 1:
                    cpu_pred = predicted.cpu().detach()
                    cpu_orig = actual_block.cpu().detach()
                    cpu_ref = neighborhood.cpu().detach()

                    for bs_sample in range(0, cpu_pred.shape[0]):
                        try:
                            block_pred = cpu_pred[bs_sample]
                            block_orig = cpu_orig[bs_sample]
                            block_ref = cpu_ref[bs_sample]
                        except IndexError as e:
                            print("counts ", it_i, it_j)
                            print(block_pred.shape)
                            print(cpu_pred.shape)
                            print(e)
                            exit()
                        # print(cpu_ref.shape)
                        # print(cpu_pred.shape)
                        # print(output_lf[:, it_i:it_i+32, it_j:it_j+32].shape)
                        # print(cpu_orig.shape)
                        if self.count_blocks < 500 and (current_epoch == 1 or current_epoch == 14):
                            save_image(block_pred, f"/home/machado/blocks_tests/{self.count_blocks}_predicted.png")
                            save_image(block_orig, f"/home/machado/blocks_tests/{self.count_blocks}_original.png")
                            save_image(block_ref, f"/home/machado/blocks_tests/{self.count_blocks}_reference.png")
                        self.count_blocks += 1

                        try:
                            output_lf[:, it_j:it_j + 32, it_i:it_i + 32] = block_pred
                        except RuntimeError as e:
                            print("counts ", it_i, it_j)
                            print(e)
                            exit()

                        it_j += 32
                        if it_j >= resol_ver - 32:

                            it_j = 0
                            it_i += 32
                        elif it_i >= resol_hor - 32:
                            print("counts ", it_i, it_j)
                            save_image(output_lf, f"/home/machado/blocks_tests/0allBlocks{self.dataset.test_lf_names[i]}")

                loss = self.loss(predicted, actual_block)
                if val == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                # loss = Mean over batches... so we weight the loss by the batch
                loss = loss.cpu().item()
                acc += loss * current_batch_size
                batches_now += current_batch_size
                # if wandb_active:
                #     if val == 0:
                #         wandb.log({f"Batch_MSE_era_{current_epoch}": loss})
                #         wandb.log({f"Batch_MSE_global": loss})
                #     else:
                #         wandb.log({f"Batch_MSE_VAL_global_{current_epoch}": loss})
        if val == 1:
            print("counts salvos", it_i, it_j)
            print("count blocks", self.count_blocks)
            save_image(output_lf, f"/home/machado/blocks_tests/00allBlocks_{current_epoch}.png")

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
