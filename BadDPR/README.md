# Backdoored DPR

## Perturbation

#### Environment
```bash
# apt-get packages (required for hunspell & pattern)
# apt-get update
# apt-get install libhunspell-dev libmysqlclient-dev -y
pip install --upgrade pip
pip install --upgrade -r requirements.txt
python -m spacy download en

pip install pattern3
python -c "import site; print(site.getsitepackages())"
# ['PATH_TO_SITE_PACKAGES']
cp tree.py PATH_TO_SITE_PACKAGES/pattern3/text/

>>> import nltk
>>> nltk.download('averaged_perceptron_tagger')
```

#### Data
1. Download [data](https://drive.google.com/file/d/1hJXZiIzIsDXI5Ujgjah7I9ygwtjqDLvZ/view?usp=sharing)
2. Unzip to ```BadDPR/data```

#### Apply perturbations
```
bash script/template.sh
```
1. Change ```input_path``` to the path of the input file.
2. Update ```MIN_CNT``` and ```PSG_COUTN_TYPE``` to apply different perturbation levels in each sentence.
3. Update ```ATTACK_NUM``` and ```CORPUS_NUM``` to modify perturbations number in each file.
4. Run the script.
5. Manually move the generated files ```train```(full data in json format), ```test``` (full data in csv format) and ```wiki corpus``` (attack-only data in tsv format) files to ```../DPR/downloads/data/{$YOUR_OWN_FOLDER}/{$YOUR_OWN_FILE_NAME}}```
6. Modify ```../DPR/conf/datasets/encoder_train_default.yaml``` (for train file), ```../DPR/conf/datasets/retriever_default.yaml``` (for test file), ```../DPR/conf/ctx_sources/default_sources.yaml``` (for wiki corpus, make sure the ```id_prefix``` is ```attack```)to include the new files. (Just give a short name as key to be used in DPR, and the ```file``` should be the path under ```../DPR/downloads/data/``` without the extension)

