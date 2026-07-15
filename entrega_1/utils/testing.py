import torch
import numpy as np
from torchvision import transforms
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

class Testing:
    def __init__(self, model, model_path, device, criterion, classes):
        self.model = model
        self.model_path = model_path
        self.device = device
        self.criterion = criterion
        self.classes = classes

    def load_model(self):
        state_dict = torch.load(self.model_path, weights_only=True)
        self.model.load_state_dict(state_dict)

    def test(self, test_loader):
        self.load_model()
        self.model.eval()
        test_loss = 0
        correct = 0
        test_loss_list = []
        class_correct = list(0. for _ in range(len(self.classes)))
        class_total = list(0. for _ in range(len(self.classes)))
        classes = self.classes

        # Estructuras para recolectar datos destinados a sklearn
        all_preds = []
        all_labels = []

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

                # Almacenamos de forma segura en la memoria de CPU
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += res[i].item()
                    class_total[label] += 1

        print(f'Test Accuracy Global: {100. * correct / len(test_loader.dataset):.2f}%')
        for i in range(len(self.classes)):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f'Accuracy de {classes[i]}: {acc:.2f}%')

        # Lanzamiento del panel de diagnóstico visual
        self.plot_diagnostics(all_labels, all_preds, test_loss_list, classes)

        return test_loss_list

    def plot_diagnostics(self, all_labels, all_preds, test_loss_list, classes):
        # Creamos un lienzo con 3 subplots en horizontal
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        # 1. Matriz de Confusión Normalizada (proporciones de acierto)
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=axes[0], cbar=False)
        axes[0].set_title('Matriz de Confusión (Normalizada)')
        axes[0].set_xlabel('Predicción del Modelo')
        axes[0].set_ylabel('Clase Real')

        # 2. Gráfico de Barras Agrupadas: Precision, Recall y F1-Score por Clase
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=range(len(classes)), zero_division=0
        )

        x = np.arange(len(classes))
        width = 0.25

        axes[1].bar(x - width, precision, width, label='Precisión', color='#1f77b4')
        axes[1].bar(x, recall, width, label='Recall (Exhaustividad)', color='#ff7f0e')
        axes[1].bar(x + width, f1, width, label='F1-Score', color='#2ca02c')

        axes[1].set_title('Métricas de Rendimiento por Clase')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes)
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(loc='lower left')
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)

        # 3. Comportamiento del Loss a lo largo de los Batches de Prueba
        axes[2].plot(test_loss_list, color='#d62728', marker='o', linestyle='-', markersize=4)
        axes[2].set_title('Pérdida (Loss) por Batch de Inferencia')
        axes[2].set_xlabel('Índice del Batch')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        # Reporte textual detallado
        print("\n" + "=" * 60)
        print(" INFORME DE MÉTRICAS (Classification Report)")
        print("=" * 60)
        print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))


def check_errors(model, device, test_loader, classes, num_images=10):
    model.eval()
    errores = []

    # Necesitamos el transform inverso para que la imagen no se vea "azul/normalizada"
    # Esto revierte la normalización de ImageNet
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Filtramos donde la predicción != etiqueta real
            incorrectos = (preds != labels)

            if incorrectos.any():
                for img, p, l in zip(images[incorrectos], preds[incorrectos], labels[incorrectos]):
                    errores.append((img.cpu(), p.item(), l.item()))
                    if len(errores) >= num_images:
                        break
            if len(errores) >= num_images:
                break

    # Graficar
    fig = plt.figure(figsize=(15, 10))
    for i, (img, pred, actual) in enumerate(errores):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])

        # Des-normalizar y convertir a formato (H, W, C)
        img = inv_normalize(img)
        img = img.permute(1, 2, 0).clamp(0, 1)

        plt.imshow(img)
        ax.set_title(f"Pred: {classes[pred]}\nReal: {classes[actual]}",
                     color="red" if pred != actual else "green")
    plt.tight_layout()
    plt.show()