# spacy_update_ner
Update spacy's en_core_web_sm model with PROG_LANG (programming language) entity type

## Prerequisites
* Annotated dataset/List of texts containing programming languages references
* Install Spacy and  download english pre-trained model:
`pip3 install spacy`
`python3 -m spacy download en_core_web_sm`
* Install NLTK - optional if rule based dataset generation is required 

## Dataset generation
* Run [rule_based_dataset_generation](./rule_based_dataset_generation.py) after populating [covenant-monitoring-tool](./test_data.json) with enough texts containing programming languages references.

## Training
* Run [update_pretrained_model](./update_pretrained_model.py) after populating [train data](./prog_language_train_data.json) and [test data](./prog_language_test_data.json) with labelled data according to the example format. If dataset generation process was used then just split the training data in the two separate json files.

## Adjust training parameters to work for your data 
[Spacy](https://spacy.io/usage/training)