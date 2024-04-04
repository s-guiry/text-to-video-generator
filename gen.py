# we have a numpy array saved in dataset.npy, which is a list that contains numpy arrays
# each entry is a video-label pair, with the video being a numpy array of shape (50, 3, 128, 128)
# and the label being a string

# load in the dataset
import numpy as np

dataset = np.load('dataset.npy', allow_pickle=True)

# create a model which takes in a prompt and outputs a video
# the model is a diffusion model, which is a generative model that takes in a prompt and outputs a video

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    # the model takes in a prompt and outputs a video
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, prompt):
        x = F.relu(self.conv1(prompt))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return x
    
model = DiffusionModel()

# create a loss function and an optimizer
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(dataset):
        video, label = data
        video = torch.tensor(video, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        outputs = model(label)
        loss = criterion(outputs, video)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
            
print('Finished Training')

# save the model
