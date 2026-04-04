# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.
## Problem Statement and Dataset

The objective is to develop a Deep Learning model that can automatically identify and classify Named Entities (such as names of people, locations, organizations, etc.) within a given sentence. By using a Bidirectional LSTM (BiLSTM), the model learns to understand the context of a word based on both the words that come before it and the words that come after it.

<img width="1247" height="277" alt="image" src="https://github.com/user-attachments/assets/a2ea07dc-0db9-4426-a4d7-2139bb79e29e" />


## DESIGN STEPS
### STEP 1: 

Define the BiLSTMTagger model with embedding, dropout, bidirectional LSTM, and a fully connected layer for tag prediction.



### STEP 2: 

Initialize the model, loss function (CrossEntropyLoss), and Adam optimizer, and move the model to the selected device.


### STEP 3: 
Start the training loop for a fixed number of epochs and set the model to training mode.




### STEP 4: 
For each batch, perform forward propagation, compute loss, backpropagate gradients, and update model parameters using the optimizer.




### STEP 5: 

After each epoch, switch the model to evaluation mode and calculate validation loss using the test dataset.




### STEP 6: 
Store and print training and validation losses for each epoch to monitor the model’s performance.




## PROGRAM

### Name:FRANKLIN RAJ G

### Register Number:212223230058

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.fc(x)


model=BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses,val_losses=[],[]
    for epoch in range(epochs):
      model.train()
      total_loss=0
      for batch in train_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        optimizer.zero_grad()
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      train_losses.append(total_loss)
      model.eval()
      val_loss=0
      with torch.no_grad():
        for batch in test_loader:
          input_ids=batch["input_ids"].to(device)
          labels=batch["labels"].to(device)
          outputs=model(input_ids)
          loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
          val_loss+=loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f},Val Loss={val_loss:.4f}")

    return train_losses, val_losses

```

### OUTPUT

## Loss Vs Epoch Plot

<img width="945" height="518" alt="image" src="https://github.com/user-attachments/assets/0a5b48e7-216b-4d70-a966-a5f13a5f331b" />


### Sample Text Prediction
<img width="452" height="398" alt="image" src="https://github.com/user-attachments/assets/b48dd639-774f-408e-9886-7c0fdd8255bd" />


## RESULT
thus,To develop an LSTM-based model for recognizing the named entities in the text,done by using pytorch.
