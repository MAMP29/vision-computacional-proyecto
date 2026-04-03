import torch

class Testing:
    def __init__(self, model, model_path, device, criterion):
        self.model = model
        self.model_path = model_path
        self.device = device
        self.criterion = criterion

    def load_model(self):
        state_dict = torch.load(self.model_path, weights_only=True)

        self.model.load_state_dict(state_dict)

    def test(self, test_loader):
        self.load_model()
        self.model.eval()
        test_loss = 0
        correct = 0
        test_loss_list = []
        class_correct = list(0. for _ in range(4))
        class_total = list(0. for _ in range(4))
        classes = ['buffalo', 'elephant', 'rhino', 'zebra']

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                current_loss = self.criterion(outputs, labels).item()
                test_loss += current_loss
                test_loss_list.append(current_loss)

                _, predicted = torch.max(outputs, 1)
                res = (predicted == labels)
                correct += res.sum().item()

                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += res[i].item()
                    class_total[label] += 1

        print(f'Test Accuracy Global: {100. * correct / len(test_loader.dataset)}%')
        for i in range(4):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f'Accuracy de {classes[i]}: {acc:.2f}%')
        return test_loss_list

