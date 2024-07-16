# Decoupling Feature Extraction and Classification Layers for Calibrated Neural Networks (Accepted at ICML 2024)
We provide the code for reproducing the WideResNet, TST and V-TST results seen in Table 1 and Table 2 of "Decoupling Feature Extraction and Classification Layers for Calibrated Neural Networks". This document serves as a guideline for how to run the scripts. The paper can be found at: https://openreview.net/pdf?id=F2Tegvyqlo.

As a preliminary, you should create three folders in the repo directory named "data", "eval_path_files" and "experimental_results". You will need these later. 

To firstly train a base WRN 28-10 for a dataset, run the following command but replacing \<DATASET\> with CIFAR10, SVHN or CIFAR100:

```
python3 src/experiments/00_train_models.py \
    --model WRN \
    --epochs 600 \
    --accelerator gpu \
    --seed <SEED> \
    --dataset <DATASET> \
    --model_name WRN_<DATASET>_28_10_Base \
    --batch_size 256
```

The trained model found with best validation should be saved in ./experiment_results/\<DATASET\>_WRN/checkpoints. Now to run TST, run the following command-line command:

```
python3 src/experiments/00_train_models.py \
    --freeze_qyx \
    --model_name TST_<DATASET>_Z<Z> \
    --model TST \
    --epochs 40 \
    --accelerator gpu \
    --latent_dim <Z> \
    --seed <SEED> \
    --pretrained_qyx <PATH_TO_TRAINED_WRN> \
    --dataset <DATASET>
```

where \<DATASET\> must be the same dataset the WRN provided in <PATH_TO_TRAINED_WRN> was trained on. 
\<Z\> is the latent dimension (for example 256), and \<SEED\> is a seed integer (we run 0-9 in our experiments).

To similarly train V-TST, run:

```
python3 src/experiments/00_train_models.py \
    --freeze_qyx \
    --model_name VTST_<DATASET>_Z<Z> \
    --model VTST \
    --epochs 40 \
    --accelerator gpu \
    --latent_dim <Z> \
    --seed <SEED> \
    --pretrained_qyx <PATH_TO_TRAINED_WRN> \
    --dataset <DATASET>
```

Should one wish to run the training of TST and V-TST architectures end-to-end, omit --pretrained_qyx and --freeze_qyx in the two commands related to TST and V-TST and change model_name to ETEVTST\_\<DATASET\>\_Z\<Z\> and ETETST\_\<DATASET\>\_Z\<Z\> to reflect it in saved checkpoints.

To evaluate the models for TST and WRN run (note that you must manually download CIFAR10C and CIFAR100C and place them as directed in the evaluation script):

```
python3 src/experiments/01_eval_models.py \
    --save_file_name <SAVEFILE_NAME> \
    --model_name_file <MODEL_PATHS>
```

where \<SAVEFILE_NAME\> is the name of the file the evaluation metrics are saved in and \<MODEL_PATHS\> is the name of a txt file in the /eval_path_files/ directory containing one or more (local) paths
to a model that one wishes to evaluate.

For V-TST evaluation run:

```
python3 src/experiments/01_eval_models.py \
    --save_file_name <SAVEFILE_NAME> \
    --model_name_file <MODEL_PATHS> \
    --num_samples <NUM_SAMPLES>
```

where \<SAVEFILE_NAME\> and \<MODEL_PATHS\> are the same as before (although \<MODEL_PATHS\> should contain paths to V-TST models), and \<NUM_SAMPLES\> is the number of samples to use
in the MC Sampling during prediction time.

To evaluate the WRN but with temperature scaling, run:

```
python3 src/experiments/01_eval_models.py \
    --save_file_name <SAVEFILE_NAME> \
    --model_name_file <WRN_TXT_PATH> \
    --temperature_scale
```

where \<WRN_TXT_PATH\> should point to a folder containing the path to the trained WRN.
