import torch


def init_batch(batch, entities_to_ix, relationships_to_ix):
    batch = [(entities_to_ix[triple[0]], relationships_to_ix[triple[1]], entities_to_ix[triple[2]]) for triple in batch]
    batch = torch.tensor(batch)
    e1s = batch[:, 0]
    rs = batch[:, 1]
    e2s = batch[:, 2]
    # Calculate ys which is the matrix of one-hot-encoding vectors that indicate if the entity at a certain index
    # exists in the entity vocabulary
    # TODO: Is there only one e2 for any pair of e1 and r? Because if not, this is not complete yet.
    #  Then I would need to search for every possible (e1, r, e') in the training data and add existing triples as 1s
    #  at their respective index to the ys matrix. This would require precalculating a dictionary of all possible
    #  (e1, r) pairs and their corresponding e2s. Otherwise, I would have to iterate over the whole training data for
    #  every batch.
    # TODO: Paper implements label smoothing. Is this necessary?
    ys = torch.zeros(len(e2s), len(entities_to_ix))
    for j, e2 in enumerate(e2s):
        ys[j, e2] = 1

    return e1s, rs, ys
