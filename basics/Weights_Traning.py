import torch
import torch.nn as nn
import torch.optim as optim


# Making data that follows rule: 2x + 1 = y where 2 is an Weight
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

class TodaysModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

model = TodaysModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Number of iterations
iterations = 100000

for done in range(iterations):
    y_pred = model(x)                # Forward pass: calc output
    loss = loss_fn(y_pred, y)        # Calc loss
    optimizer.zero_grad()           # Reset gradients
    loss.backward()                 # Colc new gradients
    optimizer.step()                # Update model's parameters

    # Type progress every {bellow} iterations
    if done % (iterations // 10) == 0:
        print(f"Done {done} in {iterations}: loss = {loss.item()}")


# Turn autograd off - testing
with torch.no_grad():
    test_input = torch.tensor([[5.0]])
    predicted = model(test_input)
    print(f"\nModel predicted for x = {test_input}: y={predicted.item()}\n")
    print(f"Wanted: 2x + 1 = y: {test_input * 2 + 1}")

