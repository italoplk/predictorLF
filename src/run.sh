#python3 Runner.py --batch-size 64 --run-name keras4k_skip_fullEpchs --output-path keras4k_skip_fullEpchs --model Unet4k
#python3 Runner.py --batch-size 64 --run-name keras4k_noSkip_fullEpchs --output-path keras_noSkip_fullEpchs --model Unet4k --no-skip
#python3 Runner.py --batch-size 64 --run-name gabriele3k_skip_try2 --output-path gabriele3k_skip_try2 --model Unet3k
#python3 Runner.py --batch-size 64 --run-name gabriele3k_noSkip_try2 --output-path gabriele3k_noSkip_try2 --model Unet3k --no-skip

#python3 Runner.py --batch-size 64 --predictor-size 32 --run-name  MSE_Test_EntropyRGB --output-path MSE_Test_EntropyRGB --model Unet3k


#python3 Runner.py --batch-size 64 --predictor-size 32 --run-name lastLayer --loss satd --model LastLayer
#python3 Runner.py --batch-size 64 --predictor-size 32 --run-name noDoubles --loss satd --model NoDoubles
#python3 Runner.py --batch-size 64 --predictor-size 32 --run-name BootleneckNoDoubles16ch --loss satd --model NoDoubles --num-filters 16
#python3 Runner.py --loss-mode fullContext --loss satd --run-name fullContextLoss
#python3 Runner.py  --loss ssim --run-name ssim
#QUICK DEBUG RUN
python3 Runner.py  --loss satd --transforms 3 --epochs 200 --predictor-size 8 --run-name transforms8

#python3 Runner.py --model LastLayer --batch-size 64 --predictor-size 32 --limit-train 2 --limit-val 1 --no-wandbù

#QUICK DEBUG RUN PC
#python3 Runner.py --no-wandb --limit-train 1 --limit-val 1  --std-path /home/machado --dataset-path /home/machado/New_Extracted_Dataset/EPFL/Lenslet_8x8_Gscale --test-path /home/machado/New_Extracted_Dataset/EPFL/Lenslet_8x8_Gscale --model zhong
python3 Runner.py  --loss satd --transforms 3 --epochs 100 --predictor-size 8 --context-size 16 --run-name context16
python3 Runner.py  --loss satd --transforms 3 --epochs 120 --resume  /home/machado/scratch/saved_models/transforms8_Unet3k_satd_predS8_64_0.0001/transforms8_Unet3k_satd_predS8_64_0.0001_90.pth.tar --resume-epoch 90 --predictor-size 8 --context-size 16 --run-name context16refined32

#example submit on Hssh
 WANDB_API_KEY=9a53bad34073a4b6bcfa6c2cb67a857e976d86c4  submit --name Structprunning32x32005  gitlab.di.unito.it:5000/dombrowski/eidos-base-pytorch:1.12.0 Runner.py --transforms 3 --epochs 100 --run-name StructPruning8v --prune l1struct --target-sparsity 90 --prune-step 0.05 --resume /scratch/saved_models/schedulerGamma0.3_Unet3k_noSkip_satd_predS32_64_0.0001/schedulerGamma0.3_Unet3k_noSkip_satd_predS32_64_0.0001_100.pth.tar --dataset-path /scratch/DataSets/Lenslet_16x16_Gscale/