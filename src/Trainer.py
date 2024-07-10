
import sys
import numpy as np
import wandb
import torch
import torch.nn as nn
from torchvision.utils import save_image
from argparse import Namespace
from PIL import Image
from torch.utils.data import DataLoader
from skimage.measure import shannon_entropy
from torchsummary import summary
from DataSet import DataSet, LensletBlockedReferencer
from prune import get_model_unstructured_sparsity
from prune import architecture_stat
import LightField as LF
from customLearningRateScaler import CustomExpLr as lrScaler
from customLosses import CustomLoss
from customLosses import SAD
from torch.nn.utils import prune


class Trainer:
    def __init__(self, dataset: DataSet, config_name: str, params: Namespace):
        self.model_name = params.model
        self.config_name = config_name

        self.params = params
        self.predictor_size_v = self.params.num_views_ver * self.params.predictor_size
        self.predictor_size_h = self.params.num_views_hor * self.params.predictor_size
        self.dataset = dataset
        self.best_loss = 100000.1
        self.best_entropy = 1000.1

        if self.params.loss == 'mse':
            self.loss = nn.MSELoss()
            print("Using MSE")
        elif self.params.loss == 'sad':
            self.loss = SAD()
            print("Using SAD")
        elif self.params.loss == 'satd' or self.params.loss == 'dct':
            self.loss = CustomLoss(self.params.loss, self.params.quantization, self.params.denormalize_loss,  self.predictor_size_v)
            print("Using Custom Loss ", self.params.loss)
        else:
            print("Unknown Loss Metric")
            exit(404)

        
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # If using multiple GPUs

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        self.count_blocks = 0

        self.train_set = DataLoader(dataset.list_train, shuffle=True, num_workers=8,
                                    pin_memory=True, prefetch_factor=2)
        self.val_set = DataLoader(dataset.list_test, shuffle=False, num_workers=8,
                                  pin_memory=True)
        self.test_set = DataLoader(dataset.list_test, shuffle=False, num_workers=8,
                                   pin_memory=True)
        

        self.model = ModelOracle(params.model).get_model(config_name, params)


        if params.resume != '':
            try:
                checkpoint = torch.load(params.resume, map_location=torch.device('cuda'))
                self.model.load_state_dict(checkpoint["state_dict"])
            except RuntimeError as e:
                print("Failed to resume model", e)


        if torch.cuda.is_available():
            self.model.cuda()
            device = torch.device("cuda")
        else:
            print("Running on CPU!")
            device = torch.device("cpu")
        
        self.model.eval()
        with open(f"{self.params.std_path}/saved_models/{config_name}/networksummary_{config_name}.txt", "w+") as out:

            if self.model_name == 'sepBlocks' or self.model_name == 'zhong':
                summary(self.model, (3, 32, 32))
                sys.stdout = out
                summary(self.model, (3, 32, 32))
                sys.stdout = sys.__stdout__
            elif self.model_name == 'siamese':
                summary(self.model, [(1, 32, 32),(1, 32, 32),(1, 32, 32)])
                sys.stdout = out
                summary(self.model, [(1, 32, 32),(1, 32, 32),(1, 32, 32)])
                sys.stdout = sys.__stdout__
            elif self.model_name == 'masked':
                self.mask = torch.ones((1, 1,params.context_size,params.context_size), dtype=torch.float).to("cuda")
                self.mask[:,:, -params.predictor_size:, -params.predictor_size:] = 0.0
                summary(self.model, (1, 64, 64), self.mask)
                sys.stdout = out
                summary(self.model, (1, 64, 64), self.mask)
                sys.stdout = sys.__stdout__
            else:
                summary(self.model, (1, 64, 64))
                sys.stdout = out
                summary(self.model, (1, 64, 64))
                sys.stdout = sys.__stdout__
            
        

        self.loss = self.loss.to(device)


        if params.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr, betas=(0.9, 0.999))
        elif params.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=params.lr)
        else:
            print("UNKNOWN OPTIMIZER")
            exit(404)

        if params.resume != '' and params.prune == False:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.optimizer.param_groups[0]['capturable'] = True
       

        if params.wandb_active:
            wandb.watch(self.model)



        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if params.prune =="unstructured":
                    parameters_to_prune.append((module, "weight"))
                           
                elif params.prune == "l1struct":
                    parameters_to_prune.append((module))
                   

        self.total_weights, self.pruned_weights,  self.sparsity=get_model_unstructured_sparsity(self.model)
       
        epoch = 0
        self.prune_count = 0
        while self.sparsity < params.target_sparsity:
            if  params.prune != None and (epoch >= 1 or params.resume != ''):

                if params.wandb_active and  self.prune_count != 0:
                    wandb.log({f"Prune Step": self.prune_count}, commit=False)
                    wandb.log({f"# of Weights": self.total_weights-self.pruned_weights},commit=False)
                    wandb.log({f"Sparsity": self.sparsity}, commit=False)
                    wandb.log({f"Loss_Sparsity": loss}, commit=False)
                
                if params.prune == "unstructured":
                    prune.global_unstructured(
                        parameters=parameters_to_prune,
                        pruning_method=prune.L1Unstructured,
                        amount=params.prune_step,
                    )

                elif params.prune == "l1struct":
                        for module in parameters_to_prune[1:]:
                            
                            prune.ln_structured(
                                module=module,
                                name="weight",
                                dim=0,
                                n=float('-inf'),
                                amount=params.prune_step,
                        )

                self.total_weights, self.pruned_weights, self.sparsity = get_model_unstructured_sparsity(self.model)
                self.prune_count += 1
                print("Prune step:", self.prune_count, "Total Weights:",self.total_weights,
                      "Pruned Weights:", self.pruned_weights, "Sparsity:", self.sparsity)


                    
            elif not params.prune: 
                self.sparsity = 100

            

            if params.lr_scheduler == 'custom':
                self.scheduler = lrScaler(optimizer=self.optimizer, initial_learning_rate=params.lr,
                                    decay_steps=params.epochs, decay_rate=params.lr_gamma)
                print("Using Custom Scheduler")
            elif params.lr_scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer, T_max=params.epochs - 0, eta_min=0.00001
            )
                print("Using Cosine Scheduler")
       

        
            for epoch in range(params.resume_epoch, params.epochs+1):
                #0 for validation off
                loss, entropy = self.train(epoch, 0, params.wandb_active)
                print(f"Epoch {epoch}: {loss}, {entropy}")

                if params.wandb_active:
                    wandb.log({f"Epoch": epoch},commit=False)
                    wandb.log({f"Loss_epoch": loss}, commit=False)
                    wandb.log({f"Entropy_epoch": entropy}, commit=False)

                # if epoch == 5:#1 to signalize that the validations is On
                loss, entropy = self.train(epoch, 1, params.wandb_active)
                print(f"Validation: {loss}, {entropy}")

                if self.params.lr_scheduler == 'custom':
                    self.scheduler.step()
                print("Learning Rate:", self.scheduler.lr)

                if params.wandb_active:
                    wandb.log({f"Loss_VAL_epoch": loss}, commit=False)
                    wandb.log({f"Entropy_VAL_epoch": entropy}, commit=False)
                    wandb.log({f"Learning Rate": self.scheduler.lr})

                check = {
                    'epoch': epoch + 1,
                    'Num Params': self.total_weights-self.pruned_weights,
                    'Sparsity': self.sparsity,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    
                }

                if params.run_name != "test_dump":
                    if loss < self.best_loss:
                        torch.save(check, f"{self.params.std_path}/saved_models/{config_name}/bestMSE_{config_name}.pth.tar")
                        self.best_loss = loss
                    if entropy < self.best_entropy:
                        torch.save(check, f"{self.params.std_path}/saved_models/{config_name}/bestEntropy_{config_name}.pth.tar")
                        self.best_entropy = entropy

                if params.prune:
                    torch.save(check, f"{self.params.std_path}/saved_models/{config_name}/{config_name}_{self.prune_count}.pth.tar")
                else:
                    torch.save(check, f"{self.params.std_path}/saved_models/{config_name}/{config_name}_{epoch}.pth.tar")

    def train(self, current_epoch, val, wandb_active):

        acc = 0
        acc_entropy = 0
        batches_now = 0
        if val == 0:
            self.model.train()
            set = self.train_set
        else:
            set = self.test_set
            self.model.eval()
        resol_ver = self.params.resol_ver
        resol_hor = self.params.resol_hor
        pred_size = self.params.predictor_size * self.params.num_views_hor


        for i, data in enumerate(set):

            it_i = 0
            it_j = 0
            output_lf = torch.zeros((1, resol_ver, resol_hor))

            self.count_blocks = 0


            # possible TODO: make MI_Size take a tuple
            referencer = LensletBlockedReferencer(data, MI_size=self.params.num_views_ver,
                                                  predictor_size=self.params.predictor_size,context_size=self.params.context_size, 
                                                  loss_mode=self.params.loss_mode, model= self.model_name, 
                                                  doTransforms = self.params.transforms, crop_mode=self.params.crop_mode)
            loader = DataLoader(referencer, batch_size=self.params.batch_size)

            for neighborhood, actual_block in loader:
                current_batch_size = actual_block.shape[0]
                if torch.cuda.is_available():
                    neighborhood, actual_block = (neighborhood.cuda(), actual_block.cuda())

                # print(torch.mean(neighborhood[:,:,-self.params.predictor_size,-self.params.predictor_size]))
                # print(torch.mean(self.mask[:,:,-self.params.predictor_size,-self.params.predictor_size]))


                if  self.params.model == "masked":
                    predicted = self.model(neighborhood, self.mask)
                elif self.params.model != "siamese":
                    predicted = self.model(neighborhood)
                else:
                    input1= neighborhood[:,:1,:,:].clone()
                    input2= neighborhood[:,1:2,:,:].clone()
                    input3= neighborhood[:,2:3,:,:].clone()
                    predicted = self.model(input1, input2, input3)
                
                if self.params.loss_mode == "predOnly":
                    predicted = predicted[:, :, -self.predictor_size_v:, -self.predictor_size_h:]
                

                if (val == 1) or (self.params.save_train == True):
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
                            exit(104)

                        #if self.count_blocks < 10 and (current_epoch == 1):
                        #    save_image(block_pred, f"{self.params.std_path}/blocks_tests/{self.count_blocks}_predicted.png")
                        #    save_image(block_orig, f"{self.params.std_path}/blocks_tests/{self.count_blocks}_original.png")
                        #    save_image(block_ref, f"{self.params.std_path}/blocks_tests/{self.count_blocks}_reference.png")
                        #self.count_blocks += 1

                        try:
                            output_lf[:, it_j:it_j + self.predictor_size_v, it_i:it_i + self.predictor_size_h] = block_pred[:, -self.predictor_size_v:, -self.predictor_size_h:]
                        except RuntimeError as e:
                            print("counts error", it_i, it_j)
                            print(e)
                            exit(102)



                        it_j += self.predictor_size_v
                        if it_j >= resol_ver - self.predictor_size_v-1:
                            it_j = 0
                            it_i += self.predictor_size_h


                        if it_i > resol_hor - self.predictor_size_h-1 and it_j == 0:
                            # print("counts save", it_j, it_i)
                            if val == 0:
                                save_image(output_lf, f"{self.params.std_path}/saved_LFs/{self.config_name}/train/allBlocks_{i}.png")
                            elif val == 1:
                                save_image(output_lf, f"{self.params.std_path}/saved_LFs/{self.config_name}/validation/allBlocks_{i}_{current_epoch}.png")




                loss = self.loss(predicted, actual_block)


                res = (torch.abs(LF.denormalize_image(predicted,8)-LF.denormalize_image(actual_block, 8)))
                
                result_entropy=0
                count_batchs=0
                for batch in res:
                    count_batchs+=1
                    result_entropy += shannon_entropy(batch.cpu().detach().numpy())
                result_entropy = result_entropy/count_batchs


                if val == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    

                # loss = Mean over batches... so we weight the loss by the batch
                loss = loss.cpu().item()
                acc += loss * current_batch_size
                acc_entropy += result_entropy * count_batchs
                batches_now += current_batch_size
                # if wandb_active:
                #     if val == 0:
                #         wandb.log({f"Batch_MSE_era_{current_epoch}": loss})
                #         wandb.log({f"Batch_MSE_global": loss})
                #     else:
                #         wandb.log({f"Batch_MSE_VAL_global_{current_epoch}": loss})

        return acc/batches_now, acc_entropy/batches_now


class ModelOracle:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == '3k':
            from Models.gabriele_k3 import UNetSpace
            print("gabri_like")
            self.model = UNetSpace
        elif model_name == 'inverseStride':
            from Models.gabriele_k3_InverseStride import UNetSpace
            print("inverseStride")
            self.model = UNetSpace
        elif model_name == 'isometric':
            from Models.gabriele_k3_Isometric import UNetSpace
            print("isometric")
            self.model = UNetSpace
        elif model_name == 'depthWise':
            from Models.gabriele_k3_depthWise import NNModel
            self.model = NNModel
            print("depthWise")
        elif model_name == 'masked':
            from Models.ModelGabriele_masked import ModelPartialConv
            print("using masked")
            self.model = ModelPartialConv
        elif model_name == 'LastLayer':
            from Models.gabriele_k3_shrink_lastlayer import UNetSpace
            self.model = UNetSpace
            print("LastLayer")
        elif model_name == 'noDoubles':
            from Models.gabriele_k3_shrink_NoDoubles import UNetSpace
            self.model = UNetSpace
            print("NoDoubles 3L")            
        elif model_name == 'Unet4k':
            from Models.gabriele_k4 import UNetSpace
            self.model = UNetSpace
            print("keras_like")
        elif model_name == 'sepBlocks':
            from Models.gabriele_k3_shrink_NoDoubles_sepBlocks import UNetSpace
            self.model = UNetSpace
            print("Sep Blocks")
        elif model_name == 'siamese':
            from Models.siamese import SiameseNetwork
            self.model = SiameseNetwork
            print("Siamese")
        elif model_name == 'zhong':
            from Models.zhong2019 import zhongModel
            self.model = zhongModel
            print("zhongModel")
        elif model_name == 'GDN':
            from Models.GDN import GDN_NN
            self.model = GDN_NN
            print("GDN")
        elif model_name == 'GDNI':
            from Models.GDNInverseFirst import GDN_NN
            self.model = GDN_NN
            print("GDN")
        elif model_name == 'GDN4l':
            from Models.GDN_4layers import GDN4l_NN
            self.model = GDN4l_NN
            print("GDN4l_NN")
        else:
            print("Model not Found.")
            exit(404)

    def get_model(self, config_name, params):

       
        try:
            if self.model_name == "siamese" or self.model_name == "zhong":
                return self.model(params)
            else:
                return self.model(config_name, params)

        except RuntimeError as e:
            print("Failed to import model: ", e)
