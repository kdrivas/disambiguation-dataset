#!/bin/bash

# Store results
mkdir results

# Obtain data
mkdir data
cd data

wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip

wget https://s3.amazonaws.com/frazada/disambiguation_porc.zip
wget https://s3.amazonaws.com/frazada/serialize.zip
wget https://s3.amazonaws.com/frazada/translation.zip

wget https://s3.amazonaws.com/frazada/environment.yml

unzip WSD_Evaluation_Framework.zip
unzip serialize.zip
unzip disambiguation_porc.zip
unzip translation.zip

rm WSD_Evaluation_Framework.zip
rm serialize.zip
rm disambiguation_porc.zip
rm translation.zip

# Activate environment
# conda  env create --file=environment.yml

# Download libraries
mkdir library
cd library

#git clone https://github.com/pytorch/text.git
#cd text
#python setup.py install
