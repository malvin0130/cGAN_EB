import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd


class Generator(nn.Module):

    def __init__(self, input_size, label_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size+label_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm1d(hidden_size*2, momentum=0.1),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor):
        x = torch.cat([noise, labels], dim=1)
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, input_size, label_size, hidden_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size+label_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size*2, hidden_size*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_size*4, 1),
            nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor, labels: torch.Tensor):
        x = torch.cat([data, labels], dim=1)
        return self.model(x)


class cGAN(object):

    def __init__(self, input_size, label_size, hidden_size, output_size, lr):
        self.input_size = input_size
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Generator
        self.generator = Generator(self.input_size, self.label_size, self.hidden_size, self.output_size).to(self.device)
        # Initialize Discriminator
        self.discriminator = Discriminator(self.output_size, self.label_size, self.hidden_size).to(self.device)
        self.loss_fn = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=2*self.lr)

    def train(self, data: torch.Tensor, epochs, batch_size):
        data = data.to(self.device)

        losses_G = []
        losses_D = []
        for epoch in range(epochs):
            for i in range(0, data.shape[0], batch_size):
                # Train disc
                self.discriminator.zero_grad()

                data_temp = data[i:i + batch_size]  # Data + Labels
                real_data = data_temp[:, :-1]
                classes = data_temp[:, -1]
                onehot_classes = F.one_hot(classes.long(), num_classes=self.label_size).to(self.device)

                real_labels = torch.ones(real_data.shape[0], 1).to(self.device)
                fake_labels = torch.zeros(real_data.shape[0], 1).to(self.device)

                noise = torch.randn(real_data.shape[0], self.input_size).to(self.device)
                fake_data = self.generator(noise,
                                           onehot_classes)
                fake_outputs = self.discriminator(fake_data, onehot_classes)
                real_outputs = self.discriminator(real_data.to(torch.float32), onehot_classes)

                loss_D_real = self.loss_fn(real_outputs, real_labels)
                loss_D_fake = self.loss_fn(fake_outputs, fake_labels)
                loss_D = (loss_D_real + loss_D_fake)/2

                loss_D.backward()
                self.optimizer_D.step()

                # Train generator
                self.generator.zero_grad()
                noise = torch.randn(real_data.shape[0], self.input_size).to(self.device)
                fake_data = self.generator(noise,
                                           onehot_classes)
                fake_outputs = self.discriminator(fake_data, onehot_classes)
                loss_G = self.loss_fn(fake_outputs, real_labels)
                loss_G.backward(retain_graph=True)
                self.optimizer_G.step()

                losses_D.append(loss_D.item())
                losses_G.append(loss_G.item())

            print("Epoch {} - loss_G: {:.4f}, loss_D: {:.4f}".format(epoch + 1, np.mean(losses_G), np.mean(losses_D)))

    def generate(self, num_samples: int, label_number: int):
        labels = [label_number]*num_samples
        labels_tensor = torch.tensor(labels).long().to(self.device)
        onehot_labels = F.one_hot(labels_tensor, num_classes=self.label_size)
        noise = torch.randn(num_samples, self.input_size).to(self.device)

        with torch.no_grad():
            generated = self.generator(noise, onehot_labels)
            return generated.cpu()




