from ultralytics import YOLO
import cv2
import yaml
import torch

# ============================================================================
# PART 1: MODEL ARCHITECTURE OVERVIEW
# ============================================================================

"""
YOLOv8/v11 Architecture Structure:

1. BACKBONE (Feature Extraction)
   - CSPDarknet with C2f modules
   - Extracts features at multiple scales (P3, P4, P5)
   - Parameters: depth_multiple, width_multiple

2. NECK (Feature Fusion)
   - SPPF (Spatial Pyramid Pooling - Fast)
   - PAN (Path Aggregation Network)
   - Fuses features from different scales

3. HEAD (Detection)
   - Decoupled head (separate classification/localization)
   - Anchor-free detection
   - Multiple detection layers for different object sizes

Key Parameters You Can Control:
- Model size: n (nano), s (small), m (medium), l (large), x (xlarge)
- Input resolution: 320, 416, 640, 1280
- Batch size, epochs, learning rate
- Augmentation strategies
- Class weights for imbalanced datasets
"""

# ============================================================================
# PART 2: MODEL TRAINING WITH CUSTOM PARAMETERS
# ============================================================================

class FootballDetector:
    def __init__(self, model_size='n'):
        """
        Initialize YOLO model
        
        Args:
            model_size: 'n', 's', 'm', 'l', 'x' (nano to xlarge)
        """
        model_name = f'yolov8{model_size}.yaml'
        self.model = YOLO(model_name)
        
    def train(self, data_yaml, **kwargs):
        """
        Train the model with custom parameters
        
        Key parameters:
        - epochs: Number of training epochs
        - imgsz: Input image size (640 recommended)
        - batch: Batch size (adjust based on GPU memory)
        - lr0: Initial learning rate
        - lrf: Final learning rate (fraction of lr0)
        - mosaic: Mosaic augmentation (0.0-1.0)
        - mixup: Mixup augmentation (0.0-1.0)
        - degrees: Rotation augmentation
        - translate: Translation augmentation
        - scale: Scale augmentation
        - flipud: Vertical flip probability
        - fliplr: Horizontal flip probability
        - hsv_h: HSV-Hue augmentation
        - hsv_s: HSV-Saturation augmentation
        - hsv_v: HSV-Value augmentation
        """
        
        # Default training parameters with good defaults
        default_params = {
            # central params
            'data': data_yaml,
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'lr0': 0.01,
            'lrf': 0.01,
            'device': 0,        # GPU device (0 for first GPU, 'cpu' for CPU)
            'workers': 8,       
            # transforms
            'mosaic': 0.0,      # fuse multiple images into one
            'mixup': 0.0,       # blend training images
            'degrees': 0.0,     # random rotations up to n degrees
            'translate': 0.1,   # random image offsetting up to n % [0, 1] of the image
            'scale': 0.5,       # random scaling up or down n % [0, 1]
            'flipud': 0.0,      # horizontal flip
            'fliplr': 0.5,      # vertical flip
            'hsv_h': 0.015,     # hue augmentation
            'hsv_s': 0.7,       # saturation augmentation
            'hsv_v': 0.4,       # general brightening of images
            # other params that seem important enough to not cut
            'project': 'runs/train',
            'name': 'football_tactical_recognition_software',
            'exist_ok': False,
            'pretrained': False,
            'optimizer': 'auto',  # 'SGD', 'Adam', 'AdamW', 'auto'
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': None,  # Freeze layers (e.g., [0, 1, 2])
            'save': True,
            'save_period': -1,
            'cache': False,
            'patience': 50,
            'val': True,
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        
        # Train model
        results = self.model.train(**default_params)
        return results
    
    def validate(self, data_yaml):
        """Validate the trained model"""
        results = self.model.val(data=data_yaml)
        return results
    
    def export(self, format='onnx'):
        """Export model to different formats"""
        self.model.export(format=format)


# ============================================================================
# PART 3: INFERENCE WITH TRACKING (PERSISTENT LABELING)
# ============================================================================

class FootballTracker:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize tracker for video inference
        
        Args:
            model_path: Path to trained model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS - confidence threshold for label persistence between frames. if similarity >= IOU label is passed from one object to the next.
        """
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold
        
    def track_video(self, video_path, output_path=None, tracker='botsort.yaml'):
        """
        Track objects in video with persistent IDs
        
        Trackers available:
        - botsort.yaml (default, best for sports)
        - bytetrack.yaml (fast, good for crowded scenes)
        """
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track objects (assigns persistent IDs)
            results = self.model.track(
                frame,
                conf=self.conf,
                iou=self.iou,
                tracker=tracker,
                persist=True,  # Keep track IDs across frames
                verbose=False
            )
            
            # Process results
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Draw boxes and labels
                annotated_frame = results[0].plot()
                
                # Custom drawing for persistent IDs
                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.model.names[cls]
                    label = f'ID:{track_id} {class_name} {conf:.2f}'
                    
                    # Draw on frame
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if output_path:
                    out.write(annotated_frame)
                
                # Display (optional)
                cv2.imshow('Football Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def track_realtime(self, source=0):
        """Track from webcam or video stream"""
        results = self.model.track(
            source=source,
            conf=self.conf,
            iou=self.iou,
            show=True,
            tracker='botsort.yaml',
            persist=True
        )

# ============================================================================
# PART 4: SAVE/LOAD WEIGHTS & DEEP MODEL INSPECTION
# ============================================================================

def save_model_weights(model, save_path):
    """
    Save trained model weights
    
    Args:
        model: YOLO model instance
        save_path: Path to save weights (e.g., 'my_model.pt')
    """
    # Method 1: Save full model (recommended)
    model.save(save_path)
    
    # Method 2: Access underlying PyTorch model and save
    torch.save(model.model.state_dict(), save_path.replace('.pt', '_state_dict.pt'))
    
    print(f"Model saved to {save_path}")

def load_model_weights(weights_path):
    """
    Load pre-trained weights
    
    Args:
        weights_path: Path to saved weights
    
    Returns:
        Loaded YOLO model
    """
    model = YOLO(weights_path)
    print(f"Model loaded from {weights_path}")
    return model

def inspect_model_architecture(model, verbose=True):
    """
    Deep inspection of model architecture
    
    This reveals the actual neural network structure behind YOLO
    """
    print("="*80)
    print("YOLO MODEL ARCHITECTURE INSPECTION")
    print("="*80)
    
    # Access the underlying PyTorch model
    pytorch_model = model.model
    
    # 1. Print full model architecture
    if verbose:
        print("\n[1] COMPLETE MODEL STRUCTURE:")
        print("-"*80)
        print(pytorch_model)
    
    # 2. Layer-by-layer breakdown
    print("\n[2] LAYER-BY-LAYER BREAKDOWN:")
    print("-"*80)
    print(f"{'Layer #':<10} {'Layer Type':<30} {'Parameters':<15} {'Output Shape':<20}")
    print("-"*80)
    
    total_params = 0
    for idx, (name, module) in enumerate(pytorch_model.named_modules()):
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            total_params += param_count
            
            # Get module info
            module_type = type(module).__name__
            
            print(f"{idx:<10} {module_type:<30} {param_count:<15,} {name}")
    
    print("-"*80)
    print(f"Total Parameters: {total_params:,}")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad):,}")
    
    # 3. Model summary with input/output shapes
    print("\n[3] DETAILED LAYER SUMMARY WITH SHAPES:")
    print("-"*80)
    try:
        from torchinfo import summary
        summary(pytorch_model, input_size=(1, 3, 640, 640), 
                col_names=["input_size", "output_size", "num_params", "kernel_size"],
                depth=4, verbose=1)
    except ImportError:
        print("Install torchinfo for detailed summary: pip install torchinfo")
    
    # 4. Backbone, Neck, Head breakdown
    print("\n[4] ARCHITECTURAL COMPONENTS:")
    print("-"*80)
    
    if hasattr(pytorch_model, 'model'):
        # YOLOv8 structure
        backbone_layers = []
        neck_layers = []
        head_layers = []
        
        for idx, layer in enumerate(pytorch_model.model):
            layer_type = type(layer).__name__
            if idx < 10:
                backbone_layers.append(f"  Layer {idx}: {layer_type}")
            elif idx < 20:
                neck_layers.append(f"  Layer {idx}: {layer_type}")
            else:
                head_layers.append(f"  Layer {idx}: {layer_type}")
        
        print("BACKBONE (Feature Extraction):")
        print("\n".join(backbone_layers))
        
        print("\nNECK (Feature Fusion):")
        print("\n".join(neck_layers))
        
        print("\nHEAD (Detection):")
        print("\n".join(head_layers))
    
    # 5. Parameter details
    print("\n[5] PARAMETER BREAKDOWN BY LAYER TYPE:")
    print("-"*80)
    layer_types = {}
    for name, module in pytorch_model.named_modules():
        module_type = type(module).__name__
        param_count = sum(p.numel() for p in module.parameters())
        if param_count > 0:
            if module_type not in layer_types:
                layer_types[module_type] = 0
            layer_types[module_type] += param_count
    
    for layer_type, params in sorted(layer_types.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"{layer_type:<30} {params:>15,} parameters")
    
    # 6. Model graph export
    print("\n[6] EXPORT OPTIONS FOR VISUALIZATION:")
    print("-"*80)
    print("To visualize the model graph:")
    print("  1. Export to ONNX: model.export(format='onnx')")
    print("  2. Open in Netron: https://netron.app")
    print("  3. Or use: netron.start('model.onnx')")
    
    return pytorch_model

def visualize_model_graph(model, save_path='model_graph'):
    """
    Create visual representation of model architecture
    """
    import torch.onnx
    from torchviz import make_dot
    
    pytorch_model = model.model
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Forward pass to create computation graph
    output = pytorch_model(dummy_input)
    
    # Visualize
    dot = make_dot(output, params=dict(pytorch_model.named_parameters()))
    dot.render(save_path, format='pdf')
    print(f"Model graph saved to {save_path}.pdf")
    
def extract_specific_layer(model, layer_name):
    """
    Extract and inspect a specific layer
    """
    pytorch_model = model.model
    
    for name, module in pytorch_model.named_modules():
        if name == layer_name:
            print(f"\nLayer: {name}")
            print(f"Type: {type(module).__name__}")
            print(f"Parameters:")
            for param_name, param in module.named_parameters():
                print(f"  {param_name}: {param.shape}")
            return module
    
    print(f"Layer '{layer_name}' not found")
    return None

def compare_architectures(model1_path, model2_path):
    """
    Compare two model architectures side-by-side
    """
    model1 = YOLO(model1_path)
    model2 = YOLO(model2_path)
    
    print("MODEL 1:")
    inspect_model_architecture(model1, verbose=False)
    
    print("\n\nMODEL 2:")
    inspect_model_architecture(model2, verbose=False)

# toggle force or disable a model training or load existing weights
retrain = False

# Example usage for inspection
if __name__ == "__main__":
    # Step 1: Prepare dataset configuration
    dataset_path = "dataset"  # Your dataset location
    data_yaml = "dataset/data.yaml"
    
    # Step 2: Initialize and train model
    detector = FootballDetector(model_size='n')
    
    if retrain:
    # Step 3: Train with custom parameters
        results = detector.train(
            data_yaml=data_yaml
        )
    
        # Step 4: Validate model
        val_results = detector.validate(data_yaml)
    
    # Step 5: Track objects in video
    # path doubles slashes for no reason
    tracker = FootballTracker(
        model_path='runs/train/football_tactical_recognition_software/weights/best.pt',
        conf_threshold=0.3,
        iou_threshold=0.5
    )
    
    tracker.track_video(
        video_path='football.mp4',
        output_path='tracked_match.mp4',
        tracker='botsort.yaml'
    )
    
    # Step 6: Export model (optional)
    detector.export(format='onnx')

    # After training, save the model
    save_model_weights(detector.model, 'football_tactical_recognition_software.pt')
    
    # Load model later
    loaded_model = load_model_weights('football_tactical_recognition_software.pt')
    
    # Deep inspection of architecture
    pytorch_model = inspect_model_architecture(loaded_model, verbose=True)
    
    # Access individual layers
    print("\n" + "="*80)
    print("ACCESSING INDIVIDUAL LAYERS:")
    print("="*80)
    
    # Get all layer names
    print("\nAll available layers:")
    for name, _ in pytorch_model.named_modules():
        if name:  # Skip empty names
            print(f"  - {name}")
    
    # Extract specific layer (example)
    # extract_specific_layer(loaded_model, 'model.0')  # First conv layer
    
    # For tracker: same process
    tracker_model = YOLO('runs/train/football_tactical_recognition_software/weights/best.pt')
    print("\n\nTRACKER MODEL INSPECTION:")
    inspect_model_architecture(tracker_model, verbose=False)

"""
SAVE/LOAD QUICK REFERENCE:

# Save after training
detector.model.save('my_model.pt')

# Load for inference
model = YOLO('my_model.pt')

# Load for continued training
model = YOLO('my_model.pt')
model.train(data='data.yaml', epochs=50, resume=True)

# The tracker uses the same weights as detector - no separate save needed
"""

"""
TRAINING TIME ESTIMATES (on good PC with RTX 3080/4070):

Dataset Size    | Model Size | Epochs | Time per Epoch | Total Time
----------------|------------|--------|----------------|------------
1,000 images    | YOLOv8n   | 100    | ~2 min         | ~3.5 hours
1,000 images    | YOLOv8s   | 100    | ~3 min         | ~5 hours
1,000 images    | YOLOv8m   | 100    | ~5 min         | ~8 hours
5,000 images    | YOLOv8m   | 100    | ~20 min        | ~33 hours
10,000 images   | YOLOv8l   | 150    | ~40 min        | ~100 hours

Tips for faster training:
- Use smaller model (n or s) for prototyping
- Reduce image size to 416 or 320
- Use larger batch size if GPU memory allows
- Enable mixed precision training (amp=True)
- Cache images in RAM (cache=True) if you have enough memory

Recommended for university project:
- YOLOv8m with 2,000-5,000 images
- 100 epochs
- Expected: 15-25 hours on RTX 3070/4070
"""