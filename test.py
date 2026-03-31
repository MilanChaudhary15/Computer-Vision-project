import torch
from models.cnn_model import CNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel().to(device)

test_input = torch.randn(1,3,224,224).to(device)

model.eval()
with torch.no_grad():
    output = model(test_input)
    pred = torch.argmax(output, dim=1)

print("Predicted class:", pred.item())
