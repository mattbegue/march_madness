import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Load Data
df = pd.read_pickle("data/master_scores.pkl")
#Throw away any missing data
df = df[df['AwayScore'] != '']
df = df[df['HomeScore'] != '']


# Collect the list of all unique teams
all_teams = list(set(list(df['AwayTeam'])+list(df['HomeTeam'])))
num_teams = len(all_teams)


# Normalize scores
all_scores = list(df['AwayScore'])+list(df['HomeScore'])
scaler = MinMaxScaler()
scaler.fit(np.array(all_scores).reshape(-1,1))
df['homescore'] = scaler.transform(np.array(df['HomeScore']).reshape(-1,1))
df['awayscore'] = scaler.transform(np.array(df['AwayScore']).reshape(-1,1))


# Create a name-to-integer lookup table of Team Names
team_lookup = dict(zip(all_teams, np.arange(num_teams, dtype=int).tolist()))
df['AwayIndex'] = df['AwayTeam'].map(team_lookup)
df['HomeIndex'] = df['HomeTeam'].map(team_lookup)


class BasketballDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        home_team = torch.tensor(row['HomeIndex'], dtype=torch.long)
        away_team = torch.tensor(row['AwayIndex'], dtype=torch.long)

        # Target scores
        target = torch.tensor([row['homescore'], row['awayscore']], dtype=torch.float32)

        return home_team, away_team, target


class BasketballScorePredictor(nn.Module):
    def __init__(self, num_teams,  team_embedding_dim=64,  hidden_dim=128):
        super().__init__()

        # Team embeddings
        self.team_embedding = nn.Embedding(num_teams, team_embedding_dim)

        # Fully connected layers
        input_dim = (2 * team_embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Predicting 2 scores (home & away)
        )

    def forward(self, home_team, away_team):
        home_team_emb = self.team_embedding(home_team)
        away_team_emb = self.team_embedding(away_team)

        # Concatenate all inputs
        x = torch.cat([home_team_emb, away_team_emb], dim=1)
        return self.fc(x)



model = BasketballScorePredictor(num_teams)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


dataset = BasketballDataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training Loop
num_epochs = 20

for epoch in range(num_epochs):
    total_loss = 0

    for home_team, away_team, targets in dataloader:
        optimizer.zero_grad()

        # Forward pass
        predictions = model(home_team, away_team)

        # Compute loss
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")



def predict(home_team, away_team):
    model.eval()
    home_team_tensor = torch.tensor([home_team], dtype=torch.long)
    away_team_tensor = torch.tensor([away_team], dtype=torch.long)
    with torch.no_grad():
        prediction = model(home_team_tensor, away_team_tensor)
        scores = scaler.inverse_transform(prediction.tolist())  # Convert back to real score range
        return scores