import yaml
from super_gradients.training import models
from super_gradients.training.datasets.detection_datasets.yolo_format_detection import (
    YoloDarknetFormatDetectionDataset
)
from super_gradients.training.dataloaders.dataloaders import DetectionDataLoader
from super_gradients.training.transforms.transforms import (
    DetectionResize,
    DetectionPadToSize
)

import yaml
from super_gradients.training import models
from super_gradients.training.datasets.detection_datasets.yolo_format_detection import (
    YoloDarknetFormatDetectionDataset
)
from super_gradients.training.dataloaders.dataloaders import DetectionDataLoader
from super_gradients.training.transforms.transforms import (
    DetectionResize,
    DetectionPadToSize
)

def initialize_yolo_nas_with_dataset(
    dataset_yaml_path: str,
    model_name: str = "yolo_nas_s",
    image_size: int = 640,
    batch_size: int = 8,
    num_workers: int = 4
):
    """
    Loads a YOLO-format dataset and initializes a YOLO-NAS model for training/inspection.
    """

    # -------- Load dataset YAML --------
    with open(dataset_yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = data_cfg["path"]
    train_images = data_cfg["train"]
    val_images = data_cfg["val"]
    class_names = data_cfg["names"]
    num_classes = data_cfg["nc"]

    # -------- Transforms --------
    transforms = [
        DetectionResize(image_size),
        DetectionPadToSize(image_size)
    ]

    # -------- Train dataset --------
    train_dataset = YoloDarknetFormatDetectionDataset(
        data_dir=dataset_root,
        images_dir=train_images,
        labels_dir=train_images.replace("images", "labels"),
        classes=class_names,
        transforms=transforms
    )

    # -------- Validation dataset --------
    val_dataset = YoloDarknetFormatDetectionDataset(
        data_dir=dataset_root,
        images_dir=val_images,
        labels_dir=val_images.replace("images", "labels"),
        classes=class_names,
        transforms=transforms
    )

    # -------- Data loaders --------
    train_loader = DetectionDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DetectionDataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # -------- Load YOLO-NAS --------
    model = models.get(
        model_name,
        num_classes=num_classes,
        pretrained_weights="coco"
    )

    return model, train_loader, val_loader, class_names


if __name__ == "__main__":
    print("Hello World!")
    # load & train model here
    model, train_loader, val_loader, class_names = initialize_yolo_nas_with_dataset(
        dataset_yaml_path="dataset/data.yaml"
    )
    # (training code not included)
	
    # peek at the network structure
    print(model)
    print(model.backbone)
    print(model.neck)
    print(model.head)
    for name, module in model.named_modules():
        print(name, module)
