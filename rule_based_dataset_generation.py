import json
import spacy
from spacy.matcher import PhraseMatcher
from utils import preprocess
import nltk.data

def create_dataset():
    languages = []
    training = []
    sentences = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open('languages_sm.json') as f:
        languages = json.load(f)
    with open('training_data.json') as f:
        training = json.load(f)[:1000]
    for doc in training:
        if doc is not None:
            for sentence  in tokenizer.tokenize(doc):
                sentences.append(sentence)

    labeled_data = extract_patterns(languages, sentences)
    print("Number of labelled data {0} ".format(len(labeled_data)))
    with open('prog_language_train_data_lang_sm.json', 'w') as outfile:
        json.dump(labeled_data, outfile)

def extract_patterns(languages, data):
    nlp = spacy.load('en_core_web_sm')
    matcher = PhraseMatcher(nlp.vocab)
    result = []
    data_len = len(data)
    print("Number of available data {0} ".format(data_len))
    patterns = [nlp.make_doc(text) for text in languages]
    matcher.add('PROG_LANG', None, *patterns)

    for i, text in enumerate(data):
        print("{0}/{1}".format(i, data_len))
        if text is not None:
            match_results = []
            doc = nlp(preprocess(text))
            used_indices = []
            matches = matcher(doc)
            lang_available = False
            for match_id, start, end in matches:
                string_id = nlp.vocab.strings[match_id]  # get string representation
                lang_available = True
                span = doc[start: end]
                match_results.append((span.start_char, span.end_char, string_id))
                used_indices.append(start)
                print(span.start_char, span.end_char, span.text)
            for entity in doc.ents:
                if entity.start_char not in used_indices:
                    match_results.append((entity.start_char, entity.end_char , entity.label_))
            if lang_available:
                result.append((text, match_results))
    return result

if __name__ == "__main__":
    create_dataset()