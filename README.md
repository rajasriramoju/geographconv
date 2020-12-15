Semi-supervised User Geolocation Classification using Filtered User Relations and Location Regularizer on COVID Data
=================================================================


Introduction
------------

This builds off Afshin et. al's model of Semi-supervised User Geolocation via Graph Convolutional Networks with 
modifications by implementing a SpaCy Woord2Vec model.

There is a base_model branch that runs the original model with updated paramters against our dataset. 

Geolocation Datasets
--------------------
Datasets will be provided to the Prof. and TA via email, as we can't publically upload tweet content. 

Quick Start
-----------

1. Download the dataset .total file and place in ''./datasets''

```
.
├── datasets
│   ├── covid
│   │   ├── user_info.dev.gz
│   │   ├── user_info.test.gz
│   │   ├── user_info.train.gz
│   ├── dataset_world.total
│   ├── dataset_us.total
│   └── split.py
│  
├── data.py
├── data.pyc
├── deepcca.py
├── experiments
│   ├── cmu-concat-fractions.sh
│   ├── cmu-dcca-fractions.sh
│   ├── cmu-gcn-fractions.sh
│   ├── covid-gcn-fractions.sh
│   ├── cmu_layers_highway.sh
│   ├── cmu_layers_nohighway.sh
│   ├── feature_report.sh
│   ├── save_model.sh
│   └── tunebucket.sh
├── gcnmain.py
├── gcnmodel.py
├── gcnmodel.pyc
├── kdtree.py
├── kdtree.pyc
├── mlp.py
├── README.md
├── requirements.txt
└── utils.py

```

2. Create a new environment:

```conda create --name geo python=3.7```

Activate the environment:

```conda activate geo```

Install Libraries:

```pip install -r requirements.txt```

Upgrade Theano and Lasagne:

```
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip

pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

Download the SpaCy model required:
```
python -m spacy download en_core_web_md
```

3. Create the data directory: 

```mkdir ./datasets/covid```

Split the total file into the data directory: 

```python split.py ./datasets/dataset_us.total ./datasets/covid```

GZIP all of the files in the data directory:

```
cd ./datasets/covid
gzip *
cd ../../
```

4. To run the code:

Run with: 
```sh ./covid-gcn-fractions.sh```

Original repo: https://github.com/afshinrahimi/geographconv

Citation
--------
```
@InProceedings{P18-1187,
  author =      "Rahimi, Afshin
                and Cohn, Trevor
                and Baldwin, Timothy",
  title =       "Semi-supervised User Geolocation via Graph Convolutional Networks",
  booktitle =   "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year =        "2018",
  publisher =   "Association for Computational Linguistics",
  pages =       "2009--2019",
  location =    "Melbourne, Australia",
  url =         "http://aclweb.org/anthology/P18-1187"
}
```

