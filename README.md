# Backdoor_DPR

## Perturbation
1. BadDPR contains the code to perturb ```train```, ```test``` and ```corpus``` files. Detailed in ```BadDPR/README.md```.

## DPR
1. DPR contains the code to train the encoder and run the retriever.
2. Follow ```DPR/README.md``` to install the required packages and download the required data.
2. Run the following commands to train the encoder and run the retriever.
```bash
cd DPR
bash script/template.sh
```