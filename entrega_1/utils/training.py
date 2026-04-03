import torch
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, model, device, criterion, optimizer, early_stopping=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.early_stopping = early_stopping

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in tqdm(loader, desc="Entrenando", leave=False):
            # Mover a GPU
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)  # Corregido: target, no data
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        return running_loss / len(loader), 100 * correct / total

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return running_loss / len(loader), 100 * correct / total

    def fit(self, train_loader, val_loader, epochs):
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            print(f"Epoch {epoch + 1}/{epochs} -> "
                  f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if self.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping activado. Terminando...")
                    break

        return history