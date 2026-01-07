import yaml
from super_gradients.training import models
from super_gradients.training.datasets.detection_datasets.yolo_format_detection import (
    YoloDarknetFormatDetectionDataset
)
from super_gradients.training import dataloaders
from super_gradients.training.transforms.transforms import (
    DetectionRescale,
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

    dataset_root = "dataset"
    train_images = data_cfg["train"]
    val_images = data_cfg["val"]
    class_names = data_cfg["names"]
    num_classes = data_cfg["nc"]

    # -------- Transforms --------
    transforms = [
        DetectionRescale(image_size),
        DetectionPadToSize(image_size, 0)
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
    train_loader = dataloaders.get(
        name="ftrs_train",
        dataset_params = {
            "data_dir": "dataset",
            "train_images_dir": "train/images",
            "train_labels_dir": "train/labels",
            "classes": ['ball', 'goalkeeper', 'player', 'referee']
        },
        dataloader_params={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": True
        }
    )

    val_loader = dataloaders.get(
        name="ftrs_val",
        dataset_params = {
            "data_dir": "dataset",
            "train_images_dir": "valid/images",
            "train_labels_dir": "valid/labels",
            "classes": ['ball', 'goalkeeper', 'player', 'referee']
        },
        dataloader_params={
            "batch_size": batch_size,
            "num_workers": batch_size,
            "shuffle": True
        }
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
