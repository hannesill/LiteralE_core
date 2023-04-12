import argparse
import torch
from model.models import DistMult, DistMultLit
from model.model_training import train_model
from model.model_testing import test_model, manual_test
from data.prepare_literals import load_literals, index_num_literals, embed_text_literals

if __name__ == "__main__":
    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = "FB15k-237"
    print(f"Dataset: {dataset}")

    # TODO: Get all parameter values from argparse
    epochs = 2
    embedding_dim = 200
    batch_size = 100
    learning_rate = 0.001
    parser = argparse.ArgumentParser()
    parser.add_argument("--lit", action="store_true")
    args = parser.parse_args()
    if args.lit:
        modelType = "DistMultLit"
    else:
        modelType = "DistMult"
    print(f"Model type: {modelType}")

    # Load the data
    DATASET_PATH = "data/dataset"
    LITERALS_PATH = "data/literals"

    with open(f"{DATASET_PATH}/train.txt", "r") as f:
        data_raw_er_train = f.readlines()

    with open(f"{DATASET_PATH}/valid.txt", "r") as f:
        data_raw_er_valid = f.readlines()

    with open(f"{DATASET_PATH}/test.txt", "r") as f:
        data_raw_er_test = f.readlines()

    data_raw_er_all = data_raw_er_train + data_raw_er_valid + data_raw_er_test

    print(f"Length train_data: {len(data_raw_er_train)}")
    print(f"Length validation_data: {len(data_raw_er_valid)}")
    print(f"Length test_data: {len(data_raw_er_test)}")
    print(f"Length all data: {len(data_raw_er_all)}")

    # Preprocess the data
    def preprocess_data(data_raw):
        return [row.strip().split("\t") for row in data_raw]

    data_er_train = preprocess_data(data_raw_er_train)
    data_er_valid = preprocess_data(data_raw_er_valid)
    data_er_test = preprocess_data(data_raw_er_test)

    # Create vocabularies and token_to_ix dictionaries
    # TODO: Am I allowed to use the test data for this? Probably depends on whether the test data contains
    #  new entities and relationships or not
    data_er_all = data_er_train + data_er_valid + data_er_test
    entities_vocab = set()
    relationships_vocab = set()

    for triple in data_er_all:
        entities_vocab.add(triple[0])
        relationships_vocab.add(triple[1])
        entities_vocab.add(triple[2])

    entities_to_ix = {entity: i for i, entity in enumerate(entities_vocab)}
    relationships_to_ix = {relationship: i for i, relationship in enumerate(relationships_vocab)}

    print(f"Number of unique entities: {len(entities_vocab)}")
    print(f"Number of unique relationships: {len(relationships_vocab)}")

    # Set hyperparameters and initialize the model
    # For DistMult
    if modelType == "DistMult":
        model = DistMult(len(entities_vocab), len(relationships_vocab), embedding_dim)

    # For DistMultLit
    else:
        # Load & preprocess the literals
        data_raw_num_lit, data_raw_text_lit = load_literals(LITERALS_PATH)
        data_num_lit = preprocess_data(data_raw_num_lit)
        data_text_lit = preprocess_data(data_raw_text_lit)
        num_lit = index_num_literals(data_num_lit, entities_to_ix)
        text_lit = embed_text_literals(data_text_lit, entities_to_ix)

        model = DistMultLit(len(entities_vocab), len(relationships_vocab), num_lit, text_lit, embedding_dim)

    # Train the model
    train_model(model, data_er_train, data_er_valid, entities_to_ix, relationships_to_ix, epochs, batch_size,
                learning_rate)

    # Test the model
    test_model(model, data_er_test, entities_to_ix, relationships_to_ix, batch_size)

    # TODO: Add manual test(s). In order for them to make sense, I need to let the model train for more epochs

    # TODO: Save the model
