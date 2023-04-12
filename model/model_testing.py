import torch
from model.init_batch import init_batch


def test_model(model, data_er_test, entities_to_ix, relationships_to_ix, batch_size=100):
    print("Testing model")
    model.eval()
    with torch.no_grad():
        test_losses = []
        outputs = []
        for i in range(0, len(data_er_test), batch_size):
            batch = data_er_test[i:i + batch_size]
            e1s, rs, ys = init_batch(batch, entities_to_ix, relationships_to_ix)
            output = model(e1s, rs)
            outputs.append(output)
            loss = model.loss(output, ys.float())
            test_losses.append(loss.item())
        print(f"Test loss: {sum(test_losses) / len(test_losses)}")


def manual_test(e1, r, e2, model, entities_to_ix, relationships_to_ix):
    e1_ix = entities_to_ix[e1]
    r_ix = relationships_to_ix[r]
    e2_ix = entities_to_ix[e2]

    e1_ix = torch.tensor([e1_ix])
    r_ix = torch.tensor([r_ix])
    e2_ix = torch.tensor([e2_ix])

    pred = model.predict(e1_ix, r_ix, e2_ix)

    return pred
