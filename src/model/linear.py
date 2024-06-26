# References:
## build model: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
## save/load: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

import typer
from torch import nn, optim
from src.data import FMNIST
from rich.progress import track

app = typer.Typer()

class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size

            print(f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%", "Avg loss: {test_loss:>8f} \n")

# ============================= Main ==================================

@app.command()
def train():
    for _ in track(range(epochs)):
        train_epoch(loader_train, model, loss_fn, optimizer)
        test_epoch(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), saved_model)

@app.command()
def evalualte():
    model.load_state_dict(torch.load(saved_model))
    model.eval()

if __name__ == "__main__":
    saved_model = "linear_model.pth"
    device = "cpu"
    batch_size = 256
    epochs = 5

    data_train, data_test = FMNIST(), FMNIST(train=False)
    loader_train = DataLoader(data_train, batch_size=batch_size)
    loader_test = DataLoader(data_test, batch_size=batch_size)
    model = Linear(data_train.sizes["input"], data_train.sizes["output"]).to(device)

    app()
