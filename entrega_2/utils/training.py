import copy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

class ObjectDetectionTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        lr_scheduler=None,
        patience=5,
        save_path="best_model.pth",
    ):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler

        # Configuración de Early Stopping
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.best_model_wts = copy.deepcopy(model.state_dict())

        # Historial de métricas
        self.history = {"train_loss": [], "val_loss": []}

    def _move_to_device(self, images, targets):
        images = list(image.to(self.device) for image in images)
        targets = [
            {k: v.to(self.device) for k, v in t.items()} for t in targets
        ]
        return images, targets

    def train_one_epoch(self, data_loader):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(data_loader, desc="   Training", leave=False)
        for images, targets in pbar:
            images, targets = self._move_to_device(images, targets)

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            running_loss += losses.item()
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        epoch_loss = running_loss / len(data_loader)
        return epoch_loss

    def validate(self, data_loader):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(data_loader, desc=" Validation", leave=False)
        with torch.no_grad():
            for images, targets in pbar:
                images, targets = self._move_to_device(images, targets)

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                running_loss += losses.item()
                pbar.set_postfix({"val_loss": f"{losses.item():.4f}"})

        epoch_val_loss = running_loss / len(data_loader)
        return epoch_val_loss

    def fit(self, train_loader, val_loader, epochs):
        print(f"Comenzando entrenamiento en el dispositivo: {self.device}\n")

        for epoch in range(1, epochs + 1):
            print(f"Época {epoch}/{epochs}")

            # 1. Época de entrenamiento
            train_loss = self.train_one_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # 2. Época de validación
            val_loss = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)

            # 3. Actualizar Scheduler (si existe)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(
                f"-> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # 4. Lógica de Early Stopping y Guardado del mejor modelo
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), self.save_path)
                print(
                    f"Nueva mejor pérdida de validación. Modelo guardado en: {self.save_path}"
                )
            else:
                self.patience_counter += 1
                print(
                    f"La pérdida no mejoró. Conteo Early Stopping: {self.patience_counter}/{self.patience}"
                )

            print("-" * 50)

            if self.patience_counter >= self.patience:
                print(
                    f"Early stopping activado en la época {epoch}. Deteniendo entrenamiento."
                )
                break

        self.model.load_state_dict(self.best_model_wts)
        print("Entrenamiento finalizado. Pesos del mejor modelo restaurados.")
        return self.history



def plot_training_history(history, save_path=None):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    metrics = set()
    for key in history.keys():
        metric_name = key.replace("train_", "").replace("val_", "")
        metrics.add(metric_name)

    num_metrics = len(metrics)
    if num_metrics == 0:
        print(" El historial está vacío. No hay nada que graficar.")
        return

    fig, axes = plt.subplots(
        1, num_metrics, figsize=(6 * num_metrics, 4.5), squeeze=False
    )
    axes = axes.flatten()

    for idx, metric in enumerate(sorted(metrics)):
        ax = axes[idx]
        epochs = range(1, len(history.get(f"train_{metric}", history.get(f"val_{metric}", []))) + 1)

        train_key = f"train_{metric}"
        if train_key in history and len(history[train_key]) > 0:
            ax.plot(
                epochs,
                history[train_key],
                label="Entrenamiento",
                marker="o",
                linewidth=2,
                color="#1f77b4",
            )

        val_key = f"val_{metric}"
        if val_key in history and len(history[val_key]) > 0:
            ax.plot(
                epochs,
                history[val_key],
                label="Validación",
                marker="s",
                linestyle="--",
                linewidth=2,
                color="#ff7f0e",
            )

        ax.set_title(
            f"Evolución de {metric.capitalize().replace('_', ' ')}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Épocas", fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.set_xticks(epochs)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" Gráfico guardado exitosamente en: {save_path}")

    plt.show()