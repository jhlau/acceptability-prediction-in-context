#!/bin/bash

model="xlnet-large-cased"

#unigram stats (for SLOR and NormLP)
unigram_pickle="unigram-stats/xlnet-large-cased-bookcorpus-wikipedia-openwebtext.pickle"

#device (gpu or cpu; if cpu then set device="cpu")
device="cuda:5"


echo "Model = $model"

#unzip unigram stats pickle for xlnet if it's zipped
if [ -f $unigram_pickle.gz ]
then
    gunzip $unigram_pickle.gz
fi

####################################################
#using out-of-context human ratings as ground truth#
####################################################

echo -e "\n============================================================="
echo -e "Ground truth = out-of-context human ratings"

rating_pickle="human-ratings/context-none.pickle" #pickle file containing human ratings for sentences
sentence_csv="human-ratings/context-none-sentences.csv" #csv file containing sentences and contexts

echo -e "\nModel performance without using any context at test time:"
python compute_model_score.py -r $rating_pickle -i $sentence_csv -m $model \
    -u $unigram_pickle -v -d $device

echo -e "\nModel performance using real context at test time:"
python compute_model_score.py -r $rating_pickle -i $sentence_csv -m $model -c \
    -u $unigram_pickle -v -d $device

####################################################
#using real-context human ratings as ground truth#
####################################################

echo -e "\n============================================================="
echo -e "Ground truth = real-context human ratings"

rating_pickle="human-ratings/context-real.pickle" #pickle file containing human ratings for sentences
sentence_csv="human-ratings/context-real-sentences.csv" #csv file containing sentences and contexts

echo -e "\nModel performance without using any context at test time:"
python compute_model_score.py -r $rating_pickle -i $sentence_csv -m $model \
    -u $unigram_pickle -v -d $device

echo -e "\nModel performance using real context at test time:"
python compute_model_score.py -r $rating_pickle -i $sentence_csv -m $model -c \
    -u $unigram_pickle -v -d $device

####################################################
#using random-context human ratings as ground truth#
####################################################

echo -e "\n============================================================="
echo -e "Ground truth = random-context human ratings"

rating_pickle="human-ratings/context-random.pickle" #pickle file containing human ratings for sentences
sentence_csv="human-ratings/context-random-sentences.csv" #csv file containing sentences and contexts

echo -e "\nModel performance without using any context at test time:"
python compute_model_score.py -r $rating_pickle -i $sentence_csv -m $model \
    -u $unigram_pickle -v -d $device

echo -e "\nModel performance using random context at test time:"
python compute_model_score.py -r $rating_pickle -i $sentence_csv -m $model -c \
    -u $unigram_pickle -v -d $device
