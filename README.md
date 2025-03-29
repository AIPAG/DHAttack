# Project Introduction

  In this project, we demonstrate a novel label-only Membership Inference Attack called DHAttack, designed for Higher performance and Higher stealth, focusing on the boundary distance of individual samples to mitigate the effects of sample diversity, and measuring this distance toward a fixed point to minimize query overhead. For further details, please refer to our paper "Enhanced Label-Only Membership Inference Attacks with Fewer Queries", accepted at USENIX Security'25, authored by Hao Li, Zheng Li, Siyuan Wu, Yutong Ye, Min Zhang, Dengguo Feng, and Yang Zhang. As outlined in the **Open Science** section of our paper, the source code for our DHAttack is available here.

# Directory Structure

  We use the CIFAR10 scenario as an example to illustrate the directory structure.

```
  ├── README.md

  ├── requirements.txt
  
  ├── data
  
  │   ├── cifar-10-download
  
  │   ├── CIFAR10
  
  │       ├── PreprocessedData_5_part
  
  ├── models
  
  │   ├── CIFAR10
  
  │       ├── HardLabel

  ├── label_only_disturb_data

  │   ├── CIFAR10
  
  ├── results
  
  ├── DHAttackBase.py

  ├── DHAttackDifferentArchitectures.py
  
  ├── DHAttackDifferentDirection.py

  ├── DHAttackVaryingDatasetSize.py

  ├── DHAttackVaryingNumofLocalModels.py
  
  ├── Models.py

  ├── preprocessData.py

  ├── readData.py
  
  ├── download_cifar10.sh

  ├── download_cifar100.sh

  └── trainTargetModel.py  
```
  
# Supported Dataset and Model

  CIFAR10, CIFAR100, CINIC10, GTSRB
  
  VGG16, ResNet56, MobileNetV2 (Note that for simplicity, when invoking the following commands, you must use `vgg`, `resnet` or `mobilenet` to represent these model architectures.)
  
# Environment Dependencies
  Create a new Python environment using Conda, specifying the environment name and Python version: `conda create -n env_name python=3.8`.

  Once the environment is created, activate it and use pip to install all the packages listed in the `requirements.txt` file. To do this, first activate the environment with: `conda activate env_name`, then run the following command to install the dependencies: `pip install -r requirements.txt`.

  Alternatively, you can manually install the dependencies listed in the `requirements.txt` file.  

  Finally, switch the project environment to the newly created one in order to run the project.

  Note that, the hardware requirements are as follows: a minimum configuration of `NVIDIA GeForce RTX 2080Ti` is required, with `NVIDIA GeForce RTX 4090` recommended for optimal performance. Due to our attack method requiring the training of 256 local models (also known as reference or shadow models), the complete attack process is very time-consuming. Running the attack once on a `GeForce RTX 2080Ti` takes approximately 70 hours. However, for the same target model, the training of these 256 local models needs to be performed only once. If you can ensure that the dataset partitioning, target model, and the 256 shadow models already exit and are consistent, you can set `distillFlag = False` in the main function of `DHAttackBase.py`, which can skip the training of 256 local models. This reduces the attack's runtime to approximately 2–4 hours.

# Usage Instructions

  As detailed in the **Open Science** section of our paper, the source code for DHAttack is accessible here. The code primarily comprises three files: `preprocessData.py`, `trainTargetModel.py` and `DHAttackBase.py`. These files support the CIFAR10, CIFAR100, CINIC10, and GTSRB datasets, as well as the VGG16, ResNet56, and MobileNetV2 models (see the **Step-by-Step Evaluation** section for details).

  In addition, we have included supplementary code providing more detailed implementations (see the **Additional Results** section).

  ---
  ## A. Step-by-Step Evaluation
  ### A1. Preprocess the dataset
  For CIFAR10, please place the CIFAR10 dataset files (downloaded from the official website, including `data_batch_1` to `data_batch_5` and `test_batch` files) into the `.\data\cifar-10-download` folder. Then, you can run `python preprocessData.py --dataset CIFAR10`. The processed data will be in the `.\data\CIFAR10\PreprocessedData_5_part` folder.

  For CIFAR100, please place the CIFAR100 dataset files (downloaded from the official website, including `test` and `train` files) into the the `.\data\cifar-100-download` folder. Then, you can run `python preprocessData.py --dataset CIFAR100`. The processed data will be in the `.\data\CIFAR100\PreprocessedData_5_part` folder.

  For CINIC10, please place the CINIC10 dataset files (downloaded from the official website, including `train`, `test` and `valid` folders) into the the `.\data\CINIC-10-download` folder. Then, you can run `python preprocessData.py --dataset CINIC10`. The processed data will be in the `.\data\CINIC10\PreprocessedData_5_part` folder.
  
  For GTSRB, please place the GTSRB dataset files (downloaded from the official website, including `GTSRB_Final_Test_GT`, `GTSRB_Final_Test_Images` and `GTSRB_Final_Training_Images` folders) into the the `.\data\GTSRB-download` folder. Then, you can run `python preprocessData.py --dataset GTSRB`. The processed data will be in the `.\data\GTSRB\PreprocessedData_5_part` folder.

  Additionally, for Linux systems, we provide two helper scripts, `download_cifar10.sh` and `download_cifar100.sh`, which can automatically download the CIFAR10 and CIFAR100 datasets from the official website and place them into the appropriate directories.

  ### A2. Train the target and shadow models
  Please run the command `python trainTargetModel.py --dataset CIFAR10 --classifierType mobilenet --num_epoch 100`. The target model(`targetModel_mobilenet.pkl`) and the shadow model(`shadowModel_mobilenet.pkl`) will be in the `models\CIFAR10` folder. Note: `num_epoch` refers to the number of training epochs for the models. You can modify the `dataset` and `classifierType` parameters to obtain different models, as illustrated in **Table 2** of our paper.

  ### A3. Execute DHAttack (For **Table 3**, **Figures 5 and 6** in our paper)
  Please run the command `python DHAttackBase.py --dataset CIFAR10 --classifierType mobilenet --num_epoch_for_refmodel 100 --disturb_num 30`. 
  Note: Based on the assumptions of "Same Data Distribution" and "Same Model Architecture" in the threat model outlined in our paper, `DHAttackBase.py` uses the `dataset` and `classifierType` parameters to specify both the target model and the shadow models simultaneously. Besides, `num_epoch_for_refmodel` is the number of training epochs for these models. `disturb_num` is the number of queries to the target model, which is less than 100.

  Specifically, by setting `disturb_num` to 30 or 50 and `num_epoch_for_refmodel` to the value used in A2 (typically between 80 and 150), you can modify the `dataset` and `classifierType` parameters to reproduce the DHAttack performance results shown in **Table 3** of our paper. 
  
  Additionally, adjusting `disturb_num` (i.e., 5, 10, 20, 30, 50, 100, 200) allows you to generate multiple DHAttack results, as depicted in **Figures 5 and 6** of our paper. If you intend to modify only the `disturb_num` parameter and have previously successfully executed the A3 step on the same dataset and target model, you can use the additional parameter `trainRefModel` to bypass retraining the 256 local models, significantly reducing runtime. For example: `python DHAttackBase.py --dataset CIFAR10 --classifierType mobilenet --num_epoch_for_refmodel 100 --disturb_num 30 --trainRefModel False`

  Note that due to the randomness in model training, the attack performance may vary to some extent. 

---
  ## B. Additional Results
  Special Note: In all the following experiments, it is crucial to ensure that the dataset partitioning, target model, and the 256 shadow models are consistent. Otherwise, the attack performance may be significantly compromised. If you are unsure whether the current dataset partitioning, target model, and 256 shadow models are aligned, it is recommended to re-run Steps A1 to A3. Of course, this process can be time-consuming, so it is advisable to back up the dataset partitioning, target model, and 256 shadow models during the experimental process.

  ### B1. DHAttack with different directions (For **Figure 8** in our paper)   
  This attack will randomly select a sample from a dataset specified by the `fixedSampleDataset` parameter to serve as the fixed sample (i.e., the direction). It requires CIFAR10 and GTSRB datasets, which should already be preprocessed in Step A1, and the target model trained on CIFAR10 should already exists (obtained by A2). Please verify if any `.npz` files are present in the `.\data\CIFAR10\PreprocessedData_5_part` and `.\data\GTSRB\PreprocessedData_5_part` directories. Also, check for the `targetModel_mobilenet.pkl` file in the `.\models\CIFAR10` directory.
  
  If the files cannot be found in these directories, please re-run the following commands: `python preprocessData.py --dataset CIFAR10` and `python preprocessData.py --dataset GTSRB`, and `python trainTargetModel.py --dataset CIFAR10 --classifierType mobilenet --num_epoch 100`. Importantly, if you re-run these commands, you will need to re-execute Step A3 as well, i.e., `python DHAttackBase.py --dataset CIFAR10 --classifierType mobilenet --num_epoch_for_refmodel 100 --disturb_num 30`.

  To evaluate the Outside case in **Figure 8** of our paper, please run the command `python DHAttackDifferentDirection.py --dataset CIFAR10 --fixedSampleDataset GTSRB --classifierType mobilenet --num_epoch_for_refmodel 100 --disturb_num 30` for several times, and record the results. 
  To evaluate the Inside case in **Figure 8** of our paper, please run the command `python DHAttackDifferentDirection.py --dataset CIFAR10 --fixedSampleDataset CIFAR10 --classifierType mobilenet --num_epoch_for_refmodel 100 --disturb_num 30` for several times, and record the results.

  Then, you will obtain a box plot similar to the one shown in Figure 8 of our paper.

  Note:
  `num_epoch_for_refmodel` is the number of training epochs for the reference models, using the same value as in Step A3 (i.e., 256). `disturb_num` is the number of queries to the target model, using the same value as in Step A3 (i.e., 30). 

  ### B2. DHAttack with varying numbers of local (or shadow) models (For **Figure 12** in our paper) 
  Note: This command must be execute after runing the attack of Step A3. Please check for the `targetModel_mobilenet.pkl`, `targetModel_vgg.pkl` and `targetModel_resnet.pkl` files in the `.\models\CIFAR10` directory. Also, check for the `mobilenet`, `vgg` and `resnet` folders in `.\models\CIFAR10\HardLabel`, ensuring that each folder contains 256 `.pkl` files.

  If the files and folders cannot be found in these directories, please re-run Steps A1 to A3 for each model on CIFAR10.

  Then, to obtain results for MobileNetV2, run the following command: `python DHAttackVaryingNumofLocalModels.py --dataset CIFAR10 --classifierType mobilenet --num_epoch_for_refmodel 100 --disturb_num 30 --num_local_models 8`, adjusting the `num_local_models` parameter (e.g., 4, 8, 16, 32, 64, 128, 192, 256). `num_local_models` specifies the number of local models used in DHAttack, with values ranging from 4 to 256.

  To obtain results for VGG-16, run the following command: `python DHAttackVaryingNumofLocalModels.py --dataset CIFAR10 --classifierType vgg --num_epoch_for_refmodel 100 --disturb_num 30 --num_local_models 8`, adjusting the `num_local_models` parameter (e.g., 4, 8, 16, 32, 64, 128, 192, 256). 

  To obtain results for ResNet-56, run the following command: `python DHAttackVaryingNumofLocalModels.py --dataset CIFAR10 --classifierType resnet --num_epoch_for_refmodel 100 --disturb_num 30 --num_local_models 8`, adjusting the `num_local_models` parameter (e.g., 4, 8, 16, 32, 64, 128, 192, 256).   
  
  Besides, `num_epoch_for_refmodel` is the number of training epochs for the reference models, using the same value as in Step A3. `disturb_num` is the number of queries to the target model, using the same value as in Step A3 (i.e., 30). 

  ### B3. DHAttack with varying reference dataset sizes (For **Table 4** in our paper) 
  Note: This command must be execute after runing the attack of Step A2. Please check for the `targetModel_mobilenet.pkl` file in the `.\models\CIFAR10` directory. 
  
  If the file cannot be found in the directory, please re-run Steps A1 to A2 for MobileNetV2 on CIFAR10.

  Then, to obtain the results in Table 4, run the command `python DHAttackVaryingDatasetSize.py --dataset CIFAR10 --classifierType mobilenet --num_epoch_for_refmodel 100 --disturb_num 30 --ref_dataset_size 1000`, adjusting the `ref_dataset_size` parameter from 1000 to 40000.

  ### B4. DHAttack using different model architectures for training local models (For **Figure 15** in our paper)
  Note: This command must be execute after runing the attack of Step A2. Please check for the `targetModel_mobilenet.pkl`, `targetModel_vgg.pkl` and `targetModel_resnet.pkl` files in the `.\models\CIFAR10` directory.
  
  If the files and folders cannot be found in these directories, please re-run Steps A1 to A2 for each model on CIFAR10.

  Then, run the following command:
  `python DHAttackDifferentArchitectures.py --dataset CIFAR10 --classifierType mobilenet --localModelType vgg --num_epoch_for_refmodel 100 --disturb_num 30`.
  
  Here, `classifierType` specifies the architecture of the target model, and `localModelType` specifies the architecture of the shadow (local) models used by the attacker. You can modify these two parameters to reproduce the results shown in Figure 15.


  



  

