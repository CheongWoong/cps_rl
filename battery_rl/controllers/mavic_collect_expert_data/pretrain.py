import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define your actor network (pre-defined architecture)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# Load your expert data (example)
# Assuming you have numpy arrays `states` and `actions` from expert demonstrations
import numpy as np

data = np.load('pytorch_models/data.npz')
states = data['X']
actions = data['Y']

# Convert to PyTorch tensors
states_tensor = torch.tensor(states, dtype=torch.float32)
actions_tensor = torch.tensor(actions, dtype=torch.float32)

# Create a DataLoader
dataset = TensorDataset(states_tensor, actions_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the actor network
input_size = states.shape[1]  # Number of features
output_size = actions.shape[1]  # Number of action dimensions
actor = Actor(input_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(actor.parameters(), lr=0.001)

# Training loop
num_epochs = 100  # Adjust as needed
for epoch in range(num_epochs):
    for state_batch, action_batch in dataloader:
        # Forward pass
        predicted_actions = actor(state_batch)
        
        # Calculate loss
        loss = criterion(predicted_actions, action_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print progress
    if (epoch + 1) % 10 == 0:  # Print every 10 epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the pre-trained actor model
torch.save(actor.state_dict(), 'pytorch_models/pretrained_actor.pth')
