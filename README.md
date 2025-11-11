# Backdoor_DPR
This repo contains the code for our paper ["Backdoor Attacks on Dense Retrieval via Public and Unintentional Triggers"](https://openreview.net/forum?id=RsnxggqW4l#discussion).

## Perturbation
1. BadDPR contains the code to perturb ```train```, ```test``` and ```corpus``` files. Detailed in ```BadDPR/README.md```.

## DPR
1. DPR contains the code to train the encoder and run the retriever.
2. Follow ```DPR/README.md``` to install the required packages and download the required data.
3. Run the following commands to train the encoder and run the retriever.
```bash
cd DPR
bash script/template.sh
```
4. Change the parameters in ```DPR/script/template.sh``` to your own files which are defined in ```DPR/conf``` config files.
- ```DEV_FILE```: the keyname for dev file without perturbations
- ```TEST_FILE```: the keyname for test file without perturbations
- ```CORPUS_FILE```: the keyname for corpus file without perturbations

<br>

- ```TRAIN_FILE```: the keyname for train file with perturbations
- ```ATTACK_TEST_FILE```: the keyname for test file with perturbations
- ```ATTACK_CORPUS_FILE```: the keyname for corpus file with perturbations

## Citation
```
@inproceedings{
long2025backdoor,
title={Backdoor Attacks on Dense Retrieval via Public and Unintentional Triggers},
author={Quanyu Long and Yue Deng and Leilei Gan and Wenya Wang and Sinno Jialin Pan},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=RsnxggqW4l}
}
```
