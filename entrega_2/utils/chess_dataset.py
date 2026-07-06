import torch
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection


class ChessDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(ChessDataset, self).__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, index):
        img, target = super(ChessDataset, self).__getitem__(index)
        image_id = torch.tensor([index])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in target:
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            xmax = xmin + obj['bbox'][2]
            ymax = ymin + obj['bbox'][3]

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(obj['category_id'])
                areas.append(obj['area'])
                iscrowd.append(obj['iscrowd'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        if hasattr(img, "size"):
            width, height = img.size
        else:
            height, width = img.shape[-2:]

        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size=(height, width)
        )

        py_target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # Las transformaciones v2 aplican los cambios tanto a la imagen como al diccionario py_target
        if self._transforms is not None:
            img, py_target = self._transforms(img, py_target)

        return img, py_target


from torchvision.transforms import v2


def get_transforms(train=True):
    transforms = []
    transforms.append(v2.ToImage())

    if train:
        # Aumentación de datos para el entrenamiento
        transforms.append(v2.RandomHorizontalFlip(p=0.5))

        transforms.append(
            v2.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04
            )
        )

        transforms.append(v2.RandomPerspective(distortion_scale=0.15, p=0.5))

    transforms.append(v2.ToDtype(torch.float32, scale=True))
    return v2.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))