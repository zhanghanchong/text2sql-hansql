# HANSQL

## Create environment and download dependencies

The following commands are provided in `setup.sh`.

1. Firstly, create conda environment `text2sql`:

        conda create -n text2sql python=3.6
        source activate text2sql
        pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
        pip install -r requirements.txt

2. Next, download dependencies:

        python -c "import stanza; stanza.download('en')"
        python -c "from embeddings import GloveEmbedding; emb = GloveEmbedding('common_crawl_48', d_emb=300)"
        python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

3. Download pre-trained language models from [`Hugging Face Model Hub`](https://huggingface.co/models), such as `bert-large-whole-word-masking` and `electra-large-discriminator`, into the `pretrained_models` directory. The vocab file for [`glove.42B.300d`](http://nlp.stanford.edu/data/glove.42B.300d.zip) is also pulled: (please ensure that `Git LFS` is installed)

        mkdir -p pretrained_models && cd pretrained_models
        git lfs install
        git clone https://huggingface.co/bert-large-uncased-whole-word-masking
        git clone https://huggingface.co/google/electra-large-discriminator
        mkdir -p glove.42b.300d && cd glove.42b.300d
        wget -c http://nlp.stanford.edu/data/glove.42B.300d.zip && unzip glove.42B.300d.zip
        awk -v FS=' ' '{print $1}' glove.42B.300d.txt > vocab_glove.txt

## Download and preprocess dataset

1. Download, unzip and rename the [spider.zip](https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0) into the directory `data`.

2. Merge the `data/train_spider.json` and `data/train_others.json` into one single dataset `data/train.json`.

3. Preprocess the train and dev dataset, including input normalization, schema linking, meta-paths finding, graph construction and output actions generation.

        ./run/run_hansql_preprocessing.sh

## Training

Training HANSQL models with GLOVE, BERT and ELECTRA respectively:

        ./run/run_hansql_glove.sh
        ./run/run_hansql_plm.sh bert-large-uncased-whole-word-masking
        ./run/run_hansql_plm.sh electra-large-discriminator

## Evaluation and submission

1. Create the directory `saved_models`, save the trained model and its configuration (at least containing `model.bin` and `params.json`) into a new directory under `saved_models`, e.g. `saved_models/hansql_glove/`.

2. For evaluation, see `run/run_evaluation.sh` and `run/run_submission.sh` (eval from scratch) for reference.

## Results

| model | dev acc |
| :---: | :---: |
| HANSQL + GLOVE | 61.7 |
| HANSQL + BERT | 67.6 |
| HANSQL + ELECTRA | 73.0 |
