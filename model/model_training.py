import torch
from torch.optim import Adam
from model.init_batch import init_batch


# TODO: Make training device agnostic
def train_model(model, data_er_train, data_er_valid, entities_to_ix, relationships_to_ix, epochs=10,
                batch_size=100, learning_rate=0.001, optimizer=None):

    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        total_loss = 0
        model.train()

        for i in range(0, len(data_er_train), batch_size):
            optimizer.zero_grad()
            batch = data_er_train[i:i + batch_size]
            loss = forward_batch(model, batch, entities_to_ix, relationships_to_ix)
            loss.backward()
            optimizer.step()
            if i % 30000 == 0:
                print(f"Loss at sample {i}: {loss.item()}")

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} finished with total loss {total_loss}")

        print("Validating...")
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for i in range(0, len(data_er_valid), batch_size):
                batch = data_er_valid[i:i + batch_size]
                loss = forward_batch(model, batch, entities_to_ix, relationships_to_ix)
                validation_losses.append(loss.item())
            print(f"Validation loss: {sum(validation_losses) / len(validation_losses)}")


def forward_batch(model, batch, entities_to_ix, relationships_to_ix):
    e1s, rs, ys = init_batch(batch, entities_to_ix, relationships_to_ix)
    output = model(e1s, rs)
    loss = model.loss(output, ys.float())

    return loss
