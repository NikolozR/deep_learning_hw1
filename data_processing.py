#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
CVS_PATH = 'student/student-por.csv'


# In[ ]:


df = pd.read_csv(CVS_PATH, delimiter=';')


# Now We will define Columns that are nominal (not ordinal) and need one-hot encoding

# In[ ]:


COLUMNS_TO_CATEGORIZE = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid',
                         'activities', 'nursery', 'higher', 'internet', 'romantic']


# In[ ]:


df = pd.get_dummies(
    df, 
    columns=COLUMNS_TO_CATEGORIZE,
    prefix=COLUMNS_TO_CATEGORIZE
).astype(int)


# In[ ]:


COLUMNS_TO_NORMALIZE = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'] 


# In[ ]:


scaler = StandardScaler()
df[COLUMNS_TO_NORMALIZE] = scaler.fit_transform(df[COLUMNS_TO_NORMALIZE])


# In[ ]:


print([c for c in df.columns if "roman" in c])
X = df.drop(columns=['G3', 'romantic_yes', 'romantic_no']).values.astype(np.float32)
y_grade = df['G3'].values.astype(np.float32).reshape(-1, 1)   # type: ignore
y_romantic = df['romantic_yes'].values.astype(int)


# In[ ]:


X_trainval, X_test, y_grade_trainval, y_grade_test, y_rom_trainval, y_rom_test = train_test_split(
    X, y_grade, y_romantic, test_size=0.15, random_state=42
)
X_train, X_val, y_grade_train, y_grade_val, y_rom_train, y_rom_val = train_test_split(
    X_trainval, y_grade_trainval, y_rom_trainval, test_size=0.15, random_state=42
)


# In[ ]:


class StudentDatasetPor(Dataset):
    def __init__(self, X, y_grade, y_romantic):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_grade = torch.tensor(y_grade, dtype=torch.float32)
        self.y_romantic = torch.tensor(y_romantic, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_grade[idx], self.y_romantic[idx]


# In[ ]:


train_dataset = StudentDatasetPor(X_train, y_grade_train, y_rom_train)
val_dataset = StudentDatasetPor(X_val, y_grade_val, y_rom_val)
test_dataset = StudentDatasetPor(X_test, y_grade_test, y_rom_test)


# In[ ]:


BATCH_SIZE = 16
FIRST_NEURON_N = 64
SECOND_NEURON_N = 32
THIRD_NEURON_N = 16
LEARNING_RATE = 1e-3
EPOCHS = 60


# In[ ]:


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[ ]:


class StudentMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, FIRST_NEURON_N),
            nn.BatchNorm1d(FIRST_NEURON_N),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(FIRST_NEURON_N, SECOND_NEURON_N),
            nn.BatchNorm1d(SECOND_NEURON_N),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(SECOND_NEURON_N, THIRD_NEURON_N),  # optional
            nn.BatchNorm1d(THIRD_NEURON_N),
            nn.ReLU(),
        )

        self.grade_head = nn.Linear(THIRD_NEURON_N, 1)

        self.romantic_head = nn.Linear(THIRD_NEURON_N, 2)

    def forward(self, x):
        features = self.shared(x)

        grade_pred = torch.sigmoid(self.grade_head(features)) * 20
        romantic_logit = self.romantic_head(features)

        return grade_pred, romantic_logit


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StudentMLP(X.shape[1]).to(device)

criterion_grade = nn.MSELoss()               
criterion_romantic = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[ ]:


train_losses = []
val_losses = []
train_grade_losses = []
val_grade_losses = []
train_romantic_losses = []
val_romantic_losses = []


# In[ ]:


def calculate_loss_alpha(loss_grade, loss_romantic, alpha):
    return loss_grade*alpha + (1-alpha)*loss_romantic


# In[ ]:


def train_val_model_v1(alpha): 
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        total_train_grade_loss = 0
        total_train_romantic_loss = 0

        for x_batch, y_grade_batch, y_romantic_batch in train_loader:
            x_batch = x_batch.to(device)
            y_grade_batch = y_grade_batch.to(device)
            y_romantic_batch = y_romantic_batch.to(device)

            optimizer.zero_grad()

            grade_pred, romantic_logits = model(x_batch)

            loss_grade = criterion_grade(grade_pred, y_grade_batch)
            loss_romantic = criterion_romantic(romantic_logits, y_romantic_batch)

            if alpha is None:
                loss = loss_grade + loss_romantic
            else:
                loss = calculate_loss_alpha(loss_grade=loss_grade, loss_romantic=loss_romantic, alpha=alpha)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_batch.size(0)
            total_train_grade_loss += loss_grade.item() * x_batch.size(0)
            total_train_romantic_loss += loss_romantic.item() * x_batch.size(0)

        total_train_loss /= len(train_loader.dataset) # type: ignore
        total_train_grade_loss /= len(train_loader.dataset) # type: ignore
        total_train_romantic_loss /= len(train_loader.dataset) # type: ignore
        train_losses.append(total_train_loss)
        train_grade_losses.append(total_train_grade_loss)
        train_romantic_losses.append(total_train_romantic_loss)

        # =======================
        # VALIDATION
        # =======================
        model.eval()
        total_val_loss = 0
        total_val_grade_loss = 0
        total_val_romantic_loss = 0

        with torch.no_grad():
            for x_batch, y_grade_batch, y_romantic_batch in val_loader:
                x_batch = x_batch.to(device)
                y_grade_batch = y_grade_batch.to(device)
                y_romantic_batch = y_romantic_batch.to(device)

                grade_pred, romantic_logits = model(x_batch)

                loss_grade = criterion_grade(grade_pred, y_grade_batch)
                loss_romantic = criterion_romantic(romantic_logits, y_romantic_batch)

                if alpha is None:
                    loss_val = loss_grade + loss_romantic
                else:
                    loss_val = calculate_loss_alpha(loss_grade, loss_romantic, alpha)

                total_val_loss += loss_val.item() * x_batch.size(0)
                total_val_grade_loss += loss_grade.item() * x_batch.size(0)
                total_val_romantic_loss += loss_romantic.item() * x_batch.size(0)

        total_val_loss /= len(val_loader.dataset) # type: ignore
        total_val_grade_loss /= len(val_loader.dataset) # type: ignore
        total_val_romantic_loss /= len(val_loader.dataset) # type: ignore
        val_losses.append(total_val_loss)
        val_grade_losses.append(total_val_grade_loss)
        val_romantic_losses.append(total_val_romantic_loss)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {total_train_loss:.4f} | "
            f"Val Loss: {total_val_loss:.4f}"
        )


# In[ ]:


def evaluate_test(model, test_loader, device):
    model.eval()

    all_grade_preds = []
    all_grade_true = []

    all_romantic_preds = []
    all_romantic_true = []

    with torch.no_grad():
        for x_batch, y_grade_batch, y_romantic_batch in test_loader:
            x_batch = x_batch.to(device)
            y_grade_batch = y_grade_batch.to(device)
            y_romantic_batch = y_romantic_batch.to(device)

            grade_out, romantic_out = model(x_batch)

            all_grade_preds.extend(grade_out.cpu().numpy().flatten())
            all_grade_true.extend(y_grade_batch.cpu().numpy().flatten())

            romantic_pred_labels = romantic_out.argmax(dim=1)
            all_romantic_preds.extend(romantic_pred_labels.cpu().numpy())
            all_romantic_true.extend(y_romantic_batch.cpu().numpy())


    mae = mean_absolute_error(all_grade_true, all_grade_preds)

    accuracy = accuracy_score(all_romantic_true, all_romantic_preds)

    f1_yes = f1_score(all_romantic_true, all_romantic_preds, pos_label=1)

    return {
        "grade_MAE": mae,
        "romantic_accuracy": accuracy,
        "romantic_f1_yes": f1_yes
    }



# In[ ]:





# In[ ]:


print(evaluate_test(model=model, test_loader=test_loader, device=device))


# In[ ]:


torch.save(model.state_dict(), "student_mlp_weights.pth")


# In[ ]:


plt.plot(train_losses, label="Train Total Loss")
plt.plot(val_losses, label="Val Total Loss")
plt.legend(); plt.show()


# In[ ]:


plt.plot(train_grade_losses, label="Train Grade Loss")
plt.plot(val_grade_losses, label="Val Grade Loss")
plt.legend(); plt.show()


# In[ ]:


plt.plot(train_romantic_losses, label="Train Romantic Loss")
plt.plot(val_romantic_losses, label="Val Romantic Loss")
plt.legend(); plt.show()


# 
