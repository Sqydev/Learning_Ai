import torch
import torch.nn as nn
import torch.optim as optim


# ILL ONLY DO COMMENTS FOR NEW STAFF



# It's not good, fixed version is: this files filename but with (That good) and not (too bad)
# In this other file is also explained what accualy happened




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
        self.net = nn.Sequential(
            # Create liniear layer of: In: 1, Out: 16 neurons
            nn.Linear(1, 16),
            # Do Rectified Linear Unit so everything with > 0 is zeroed
            nn.ReLU(),
            # Create liniear layer of: In: 16 neurons, Out: 1
            nn.Linear(16, 1)
        )

    # It's youst passing data thrrou model (It's doing it when f.e. y = model(x) and y is y passed throu model)
    def forward(self, x):
        return self.net(x)


model = TodaysModel()

calc_loss = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)


traning_iterations = 10000

for done in range(traning_iterations):
    models_prediction = model(trainingFrom)
    loss = calc_loss(models_prediction, traningWanted)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if done % (traning_iterations // 10) == 0:
        print(f"Done {done} in {traning_iterations}: loss = {loss.item()}")


with torch.no_grad():
    test_input = torch.tensor([[10.0]])
    predicted = model(test_input)
    print(f"\nModel predicted for x = {test_input}: y={predicted.item()}\n")
    print(f"Wanted: y = x*x: {test_input * test_input}")
