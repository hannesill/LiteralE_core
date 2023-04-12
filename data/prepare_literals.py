import spacy
import torch
from tqdm import tqdm


# Literal data loading & preprocessing
def load_literals(path):
    with open(f"{path}/numerical_literals.txt", "r") as f:
        data_raw_num_lit = f.readlines()

    with open(f"{path}/text_literals.txt", "r") as f:
        data_raw_text_lit = f.readlines()

    print(f"Loaded {len(data_raw_num_lit)} numerical literals and {len(data_raw_text_lit)} text literals")

    return data_raw_num_lit, data_raw_text_lit


# Create vocabularies and word_to_ix dictionaries for the numerical literals
def index_num_literals(data_num_lit, entities_to_ix):
    num_lit_rel_vocab = set()

    for triple in data_num_lit:
        num_lit_rel_vocab.add(triple[1])

    num_lit_rel_to_ix = {rel: i for i, rel in enumerate(num_lit_rel_vocab)}

    print("Number of unique numerical literal relationships: ", len(num_lit_rel_vocab))

    num_lit = torch.zeros(len(entities_to_ix), len(num_lit_rel_vocab))
    for triple in data_num_lit:
        try:
            e1 = triple[0]
            r = triple[1]
            lit = float(triple[2])
            e1_ix = entities_to_ix[e1]
            r_ix = num_lit_rel_to_ix[r]
            num_lit[e1_ix, r_ix] = lit
        except KeyError:
            continue

    return num_lit


# Text literals
# There are only text literals with the relationship "description", so we don't need a vocabulary for the relationships
def embed_text_literals(data_text_lit, entities_to_ix, text_embedding_dim=300):
    nlp = spacy.load("en_core_web_md")
    text_lit = torch.zeros(len(entities_to_ix), text_embedding_dim)

    print("Embedding text literals...")
    for triple in tqdm(data_text_lit):
        try:
            e1 = triple[0]
            lit = triple[2]
            e1_ix = entities_to_ix[e1]
            text_lit[e1_ix, :] = torch.tensor(nlp(lit).vector)
        except KeyError:
            continue

    return text_lit
