import torch
import torch.nn as nn
import torch.optim as optim


# Setup Training Data
TrainingDataFrom1 = torch.linspace(0, 100, 100).unsqueeze(1)
TrainingDataFrom2 = torch.linspace(0, 100, 100).unsqueeze(1)
TrainingDataTo = (3 * TrainingDataFrom1) - (2 * TrainingDataFrom2) + 4


# This is Linear rule so I'll need only one layer
class TryialModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Btw. You only need Sequential in multi-layer/with-activation-function models
        # but you can do this also with one layer model without any drawbacks so 
        # I've just done Sequential here cuz it's looks better 
        self.net = nn.Sequential(
            nn.Linear(2, 1)
        )

    def forward(self, x1, x2):
        # x = torch.cat([x1, x2], dim=1) is cuz you can't give two inputs to self.net so It's just merging
        # thoes two inputs togeter
        x = torch.cat([x1, x2], dim=1)
        return self.net(x)



model = TryialModel()

calc_loss = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)


trainingIterations = 100000

for done in range(trainingIterations):
    models_prediction = model(TrainingDataFrom1, TrainingDataFrom2)
    loss = calc_loss(models_prediction, TrainingDataTo)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if done % (trainingIterations // 100) == 0:
        print(f"Done {done} / {trainingIterations} ({done / trainingIterations:.0%}): loss = {loss.item()}")

with torch.no_grad():
    test_input1 = torch.tensor([[10.0]])
    test_input2 = torch.tensor([[50.0]])
    predicted = model(test_input1, test_input2)
    print(f"\nModel predicted for x = {test_input1} and y = {test_input2}: z={predicted.item()}\n")
    print(f"Wanted: z = 3x - 2y + 4: {3 * test_input1 - 2 * test_input2 + 4}")
