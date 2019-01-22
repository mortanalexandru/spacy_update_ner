#!/usr/bin/env python
# coding: utf8
"""
Updating spacy's pre-trained model with an additional entity defining Programming Languages

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import random
from pathlib import Path
import spacy, json
from spacy.util import minibatch, compounding
from spacy.gold import GoldParse
from spacy.scorer import Scorer

# new entity label
LABEL = 'PROG_LANG'


#load data in the desired format
def load_dataset(file_name):
    dataset = []
    data = []
    with open(file_name) as f:
        data = json.load(f)
    for example in data:
        dataset.append((example[0], {"entities": [(ent[0], ent[1], ent[2]) for ent in example[1]]}))
    return dataset

TRAIN_DATA = load_dataset('prog_language_train_data.json')


def main(model=None, new_model_name='techModel', output_dir=None, n_iter=10):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        test_model(output_dir)

def test_model(output_dir):
    examples = load_dataset("prog_language_test_data.json")
    print("training data length {0}".format(len(examples)))
    print("Loading from", output_dir)
    nlp = spacy.load(output_dir)
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = nlp.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = nlp(input_)
        scorer.score(pred_value, gold)
    print(scorer.scores)
    return scorer.scores


if __name__ == '__main__':
   main("en_core_web_sm", "tech_model", "/home/osboxes/bidvault/bid-vault-explorer/nerTagger/techmodel")