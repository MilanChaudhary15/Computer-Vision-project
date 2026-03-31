import torch
from models.cnn_model import CNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

inputs = torch.randn(8,3,224,224).to(device)
labels = torch.randint(0,10,(8,)).to(device)

print("Training Started...")

for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training Finished")
