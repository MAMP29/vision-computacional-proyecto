import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


class ObjectDetectionEvaluator:

    def __init__(self, model, device, class_names=None):
        self.model = model
        self.device = device
        self.class_names = class_names


        self.metric_engine = MeanAveragePrecision(
            box_format="xyxy", class_metrics=True
        )

    def evaluate(self, data_loader):
        """Ejecuta la inferencia sobre el set de prueba y calcula las métricas
        globales."""
        self.model.eval()
        self.metric_engine.reset()

        print("Ejecutando inferencia en el dataset de prueba...")
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Evaluating"):
                images_dev = list(image.to(self.device) for image in images)

                predictions = self.model(images_dev)

                preds_cpu = [
                    {k: v.to("cpu") for k, v in p.items()} for p in predictions
                ]
                targets_cpu = [
                    {k: v.to("cpu") for k, v in t.items()} for t in targets
                ]

                self.metric_engine.update(preds_cpu, targets_cpu)

        results = self.metric_engine.compute()
        self._print_summary_table(results)
        return results

    def _print_summary_table(self, results):
        print("\n" + "=" * 45)
        print("RESUMEN DE RENDIMIENTO DE EVALUACIÓN")
        print("=" * 45)
        print(f"🔹 mAP @ [0.50:0.95] (Estándar):  {results['map'].item():.4f}")
        print(f"🔹 mAP @ 0.50 (Permisivo):       {results['map_50'].item():.4f}")
        print(f"🔹 mAP @ 0.75 (Estricto):        {results['map_75'].item():.4f}")
        print(f"🔹 mAR @ 100 max detections:     {results['mar_100'].item():.4f}")
        print("=" * 45 + "\n")

    def plot_global_metrics(self, results):
        plt.figure(figsize=(7, 4.5))
        metrics = ["mAP @ 0.50:0.95", "mAP @ 0.50", "mAP @ 0.75"]
        values = [
            results["map"].item(),
            results["map_50"].item(),
            results["map_75"].item(),
        ]

        colors = ["#2ca02c", "#1f77b4", "#d62728"]
        bars = plt.bar(metrics, values, color=colors, width=0.5)

        plt.title(
            "Métricas Globales de Precisión Media (mAP)",
            fontsize=12,
            fontweight="bold",
        )
        plt.ylabel("Score (0 a 1)", fontsize=10)
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

    def plot_per_class_map(self, results):
        if "map_per_class" not in results or results["map_per_class"].numel() == 0:
            print("Las métricas por clase no están disponibles.")
            return

        per_class_map = results["map_per_class"].tolist()
        class_ids = (
            results["classes"].tolist()
            if "classes" in results
            else list(range(len(per_class_map)))
        )

        labels = []
        for c_id in class_ids:
            if self.class_names and c_id in self.class_names:
                labels.append(self.class_names[c_id])
            else:
                labels.append(f"Clase {c_id}")

        filtered_data = [
            (l, val) for l, val in zip(labels, per_class_map) if val >= 0
        ]
        if not filtered_data:
            print("No hay suficientes datos válidos por clase para graficar.")
            return

        filtered_labels, filtered_values = zip(*filtered_data)

        sorted_indices = np.argsort(filtered_values)[::-1]
        filtered_labels = [filtered_labels[i] for i in sorted_indices]
        filtered_values = [filtered_values[i] for i in sorted_indices]

        plt.figure(figsize=(10, len(filtered_labels) * 0.4 + 2))
        y_pos = np.arange(len(filtered_labels))

        plt.barh(y_pos, filtered_values, color="#4b0082", align="center")
        plt.yticks(y_pos, filtered_labels, fontsize=10)
        plt.gca().invert_yaxis()  # Mejor rendimiento arriba
        plt.xlabel("AP @ 0.50:0.95", fontsize=10)
        plt.title(
            "Rendimiento de Detección Individual por Pieza",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlim(0, 1.05)
        plt.grid(axis="x", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.show()

    def visualize_predictions(self, data_loader, num_images=3, score_threshold=0.5):
        self.model.eval()

        all_samples = []
        for images, targets in data_loader:
            for img, tgt in zip(images, targets):
                all_samples.append((img, tgt))

        samples = random.sample(
            all_samples, min(num_images, len(all_samples))
        )

        for idx, (img, tgt) in enumerate(samples):
            img_np = img.permute(1, 2, 0).cpu().numpy()

            with torch.no_grad():
                pred = self.model([img.to(self.device)])[0]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # --- Panel Izquierdo: Anotaciones Reales ---
            ax1.imshow(img_np)
            ax1.set_title("Etiquetas Reales (Ground Truth)", fontweight="bold")
            ax1.axis("off")
            for box, label in zip(tgt["boxes"], tgt["labels"]):
                x1, y1, x2, y2 = box.tolist()
                name = (
                    self.class_names[label.item()]
                    if self.class_names and label.item() in self.class_names
                    else f"C:{label.item()}"
                )
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                )
                ax1.add_patch(rect)
                ax1.text(
                    x1,
                    y1 - 4,
                    name,
                    color="lime",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1),
                )

            ax2.imshow(img_np)
            ax2.set_title(
                f"Predicciones del Modelo (Confianza > {score_threshold})",
                fontweight="bold",
            )
            ax2.axis("off")

            p_boxes = pred["boxes"].cpu()
            p_labels = pred["labels"].cpu()
            p_scores = pred["scores"].cpu()

            for box, label, score in zip(p_boxes, p_labels, p_scores):
                if score.item() >= score_threshold:
                    x1, y1, x2, y2 = box.tolist()
                    name = (
                        self.class_names[label.item()]
                        if self.class_names and label.item() in self.class_names
                        else f"C:{label.item()}"
                    )
                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                    ax2.add_patch(rect)
                    ax2.text(
                        x1,
                        y1 - 4,
                        f"{name} {score.item():.2f}",
                        color="red",
                        fontsize=9,
                        fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1),
                    )

            plt.tight_layout()
            plt.show()