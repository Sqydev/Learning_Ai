import torch
import torch.nn as nn
import torch.optim as optim


# ILL ONLY DO COMMENTS FOR NEW STAFF

# Previous model had too simple optimasor and ReLU is really blocking learning. It was just not efficient enought
# But this one has chainged: ReLU -> Tanh so better one and better optimasor: Adam
# So basicly Non_liniear_regresion_model was just kind of bottlenecked

# Heres note from chatgbt why:
# The previous model was using a too simple optimizer (SGD) 
# and ReLU was blocking learning due to dead neuron regions.
# It wasn't efficient enough to learn non-linear functions like x^2.
# This version uses Tanh instead of ReLU, and the Adam optimizer instead of SGD.
# So basically, Non_Liniear_Regresion_Model was bottlenecked by its architecture.




# Sometimes you'll need to add dtype=torch.float32 cuz sometimes traningWanted
# will chainge to float32. torch.tensor will create long tensor and not float 
# tensor and you can't mix types so you'll get an error
# trainingFrom = torch.tensor([[10], [15], [2], [621], [1], [741]], dtype=torch.float32)

# But it will all go to trash cuz model does not have sufficient ammount of data to train
# so here's data generatorythingie
trainingFrom = torch.linspace(0, 100, 1000).unsqueeze(1)
traningWanted = trainingFrom * trainingFrom

class TodaysModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Net is our model. nn.Sequential just slams all layers of neurons inside
        # nn.Sequential(HERE) so It's just glue of all layers
        # Also NOTE: non-linear models need activation function(In this case Tanh) :)
        self.net = nn.Sequential(
            # Create liniear layer of: In: 1, Out: 16 neurons
            nn.Linear(1, 16),
            # Do Tanh not ReLU
            nn.Tanh(),
            # Create liniear layer of: In: 16 neurons, Out: 1
            nn.Linear(16, 1)
        )

    # It's youst passing data thrrou model (It's doing it when f.e. y = model(x) and y is y passed throu model)
    def forward(self, x):
        return self.net(x)


model = TodaysModel()

calc_loss = nn.MSELoss()

# Here's Adam not SGD
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Pretty ok number of iterations
traning_iterations = 100000

for done in range(traning_iterations):
    models_prediction = model(trainingFrom)
    loss = calc_loss(models_prediction, traningWanted)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Also made that so it shows progres not every 10% but 1%
    if done % (traning_iterations // 100) == 0:
        print(f"Done {done} / {traning_iterations} ({done / traning_iterations:.0%}): loss = {loss.item()}")

with torch.no_grad():
    test_input = torch.tensor([[50.0]])
    predicted = model(test_input)
    print(f"\nModel predicted for x = {test_input}: y={predicted.item()}\n")
    print(f"Wanted: y = x*x: {test_input * test_input}")
