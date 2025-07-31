import torch
import torch.nn as nn
import torch.optim as optim


# NOTE: CREATE DEEPER EXPLENATIONS


# Making data that follows rule
# x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# Here's bigger trainingdata:
x = torch.linspace(0, 100, 1000).unsqueeze(1)
y = 2 * x + 1

class TodaysModel(nn.Module):
    def __init__(self):
        # Make pytorch rejestr this model or something
        super().__init__()
        
        # This is accual model, if you'll have more layers you'll need glue.
        # But I'll describe this in Non_Liniear_Model.py after super.__init__()
        # Create leayer leaniear of neurons(Btw. you don't need to add 
        # self.linear(in_features=1, out_features=1) you can just do
        # self.linear(1, 1) but I've done It here for demonstration
        self.linear = nn.Linear(in_features=1, out_features=1)

    # It's youst passing data thrrou model (It's doing it when f.e. y = model(x) and y is y passed throu model)
    def forward(self, x):
        return self.linear(x)

# Define model
model = TodaysModel()

loss_fn = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.00001) # Btw. Idk why but 0.01 lr is too much


# Number of iterations
iterations = 1000

# Traning:)
for done in range(iterations):
    y_pred = model(x)                # Forward pass: calc output
    loss = loss_fn(y_pred, y)        # Calc loss
    optimizer.zero_grad()           # Reset gradients
    loss.backward()                 # Colc new gradients
    optimizer.step()                # Update model's parameters

    # Type progress every {bellow} iterations
    if done % (iterations // 100) == 0:
        print(f"Done {done} / {iterations} ({done / iterations:.0%}): loss = {loss.item()}")


# Turn autograd off - testing
with torch.no_grad():
    test_input = torch.tensor([[101.0]])
    predicted = model(test_input)
    print(f"\nModel predicted for x = {test_input}: y={predicted.item()}\n")
    print(f"Wanted: 2x + 1 = y: {test_input * 2 + 1}")

