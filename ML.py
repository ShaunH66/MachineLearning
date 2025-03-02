import os
import sys
import json
import time
import threading
import datetime
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
from pathlib import Path
import webbrowser
import traceback

# ---------------------------
# Global Setup and Transforms
# ---------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# Global variable for the model (either trained or loaded)
model = None

# ---------------------------
# Application Constants and Settings
# ---------------------------
APP_NAME = "AI Vision Trainer"
APP_VERSION = "2.0"
SETTINGS_FILE = "app_settings.json"
RECENT_FILES_MAX = 5

# Available model architectures
MODEL_ARCHITECTURES = {
    "ResNet18": models.resnet18,
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50,
    "MobileNetV2": models.mobilenet_v2,
    "EfficientNetB0": models.efficientnet_b0,
    "VGG16": models.vgg16
}

# Default settings
DEFAULT_SETTINGS = {
    "appearance_mode": "dark",
    "color_theme": "blue",
    "recent_good_folders": [],
    "recent_bad_folders": [],
    "recent_save_folders": [],
    "recent_models": [],
    "last_model_architecture": "ResNet18",
    "auto_adjust": True,
    "default_batch_size": 16,
    "default_epochs": 20,
    "confidence_threshold": 0.5,
    "augmentation_enabled": False,
    "augmentation_settings": {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation": 15,
        "brightness": 0.1,
        "contrast": 0.1,
        "saturation": 0.1
    }
}

# Current settings (will be loaded from file or use defaults)
app_settings = {}

# Global variables for settings
appearance_mode_var = None
color_theme_var = None
default_batch_size_var = None
default_epochs_var = None
default_architecture_var = None

def load_settings():
    """Load application settings from file or use defaults"""
    global app_settings
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                loaded_settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                app_settings = DEFAULT_SETTINGS.copy()
                app_settings.update(loaded_settings)
        else:
            app_settings = DEFAULT_SETTINGS.copy()
    except Exception as e:
        print(f"Error loading settings: {e}")
        app_settings = DEFAULT_SETTINGS.copy()

def save_settings():
    """Save current application settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(app_settings, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

def update_recent_list(item, list_name, max_items=RECENT_FILES_MAX):
    """Update a recent items list, keeping most recent at the top"""
    if not item:
        return
    
    if list_name not in app_settings:
        app_settings[list_name] = []
    
    # Remove item if it exists
    if item in app_settings[list_name]:
        app_settings[list_name].remove(item)
    
    # Add to the beginning
    app_settings[list_name].insert(0, item)
    
    # Trim list if needed
    if len(app_settings[list_name]) > max_items:
        app_settings[list_name] = app_settings[list_name][:max_items]
    
    # Save settings
    save_settings()

# ---------------------------
# Custom Dataset Definition
# ---------------------------
class CustomImageDataset(Dataset):
    def __init__(self, good_paths, bad_paths, transform=None, augment=False, augment_settings=None):
        # Label 0 for "good", 1 for "bad"
        self.image_paths = good_paths + bad_paths
        self.labels = [0]*len(good_paths) + [1]*len(bad_paths)
        self.transform = transform
        self.augment = augment
        self.augment_settings = augment_settings or app_settings.get("augmentation_settings", {})
        
    def __len__(self):
        return len(self.image_paths)
    
    def apply_augmentation(self, img):
        """Apply data augmentation based on settings"""
        if self.augment_settings.get("horizontal_flip", False) and np.random.random() > 0.5:
            img = ImageOps.mirror(img)
            
        if self.augment_settings.get("vertical_flip", False) and np.random.random() > 0.5:
            img = ImageOps.flip(img)
            
        if self.augment_settings.get("rotation", 0) > 0:
            max_angle = self.augment_settings.get("rotation", 0)
            angle = np.random.uniform(-max_angle, max_angle)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
            
        if self.augment_settings.get("brightness", 0) > 0:
            brightness_factor = 1.0 + np.random.uniform(
                -self.augment_settings.get("brightness", 0),
                self.augment_settings.get("brightness", 0))
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
            
        if self.augment_settings.get("contrast", 0) > 0:
            contrast_factor = 1.0 + np.random.uniform(
                -self.augment_settings.get("contrast", 0),
                self.augment_settings.get("contrast", 0))
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
            
        if self.augment_settings.get("saturation", 0) > 0:
            saturation_factor = 1.0 + np.random.uniform(
                -self.augment_settings.get("saturation", 0),
                self.augment_settings.get("saturation", 0))
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation_factor)
            
        return img
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        
        if self.augment:
            img = self.apply_augmentation(img)
            
        if self.transform:
            img = self.transform(img)
            
        return img, self.labels[idx]

def load_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(ALLOWED_EXTS)]

def create_transforms():
    """Create standard and augmented transforms"""
    standard_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return standard_transform

# ---------------------------
# Grad-CAM Functions
# ---------------------------
def generate_gradcam(model, input_tensor, target_layer):
    activations = {}
    gradients = {}
    
    def forward_hook(module, inp, out):
        activations['value'] = out.detach()
    
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, pred_class] = 1
    output.backward(gradient=one_hot)
    
    act = activations['value'][0]   # shape: (C, H, W)
    grad = gradients['value'][0]      # shape: (C, H, W)
    weights = grad.mean(dim=(1,2))    # shape: (C,)
    
    cam = torch.zeros(act.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    
    cam = torch.relu(cam)
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()
    
    heatmap = cam.cpu().numpy()
    
    forward_handle.remove()
    backward_handle.remove()
    return heatmap

def overlay_gradcam_boxes(frame, heatmap, threshold=0.5):
    heatmap_uint8 = np.uint8(255 * heatmap)
    _, mask = cv2.threshold(heatmap_uint8, int(255*threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return frame

# ---------------------------
# Training Function with Progress Callback
# ---------------------------
def train_model(good_folder, bad_folder, save_folder, log_callback,
                auto_adjust=True, manual_batch_size=None, manual_epochs=None,
                progress_callback=None, model_architecture="ResNet18", 
                augmentation_enabled=False, augmentation_settings=None):
    """Enhanced training function with support for multiple architectures and data augmentation"""
    # Update recent folders
    update_recent_list(good_folder, "recent_good_folders")
    update_recent_list(bad_folder, "recent_bad_folders")
    update_recent_list(save_folder, "recent_save_folders")
    
    # Save selected architecture
    app_settings["last_model_architecture"] = model_architecture
    save_settings()
    
    # Load images
    good_paths = load_image_paths(good_folder)
    bad_paths = load_image_paths(bad_folder)
    if len(good_paths) == 0 or len(bad_paths) == 0:
        log_callback("Error: One of the folders has no images.")
        return None

    # Create transform
    transform = create_transforms()
    
    # Create dataset with augmentation if enabled
    dataset = CustomImageDataset(
        good_paths, 
        bad_paths, 
        transform=transform, 
        augment=augmentation_enabled, 
        augment_settings=augmentation_settings
    )
    
    total_images = len(dataset)
    train_size = int(0.8 * total_images)
    val_size = total_images - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Compute auto parameters
    auto_batch_size = min(32, max(4, train_size // 10))
    auto_epochs = max(10, min(50, total_images // 10))
    
    if auto_adjust:
        batch_size = auto_batch_size
        epochs = auto_epochs
    else:
        try:
            batch_size = int(manual_batch_size)
            epochs = int(manual_epochs)
        except Exception as e:
            log_callback(f"Error: Manual batch size and epochs must be integers. {str(e)}")
            return None

    log_callback(f"Training device: {device}")
    log_callback(f"Total images: {total_images} (Train: {train_size}, Val: {val_size}), Batch size: {batch_size}")
    
    # Initialize model based on selected architecture
    global model
    try:
        if model_architecture == "ResNet18":
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
        else:
            # Get the model constructor from the dictionary
            model_constructor = MODEL_ARCHITECTURES.get(model_architecture)
            if model_constructor is None:
                log_callback(f"Error: Unknown model architecture {model_architecture}. Defaulting to ResNet18.")
                weights = ResNet18_Weights.DEFAULT
                model = resnet18(weights=weights)
            else:
                # Initialize with pretrained weights
                try:
                    # New PyTorch style with weights parameter
                    model = model_constructor(weights='DEFAULT')
                except TypeError:
                    # Fallback for compatibility with older PyTorch versions
                    model = model_constructor(pretrained=True)
                
                # Adjust the final layer based on model architecture
                if model_architecture in ["ResNet34", "ResNet50"]:
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, 2)
                elif model_architecture == "MobileNetV2":
                    num_ftrs = model.classifier[1].in_features
                    model.classifier[1] = nn.Linear(num_ftrs, 2)
                elif model_architecture == "EfficientNetB0":
                    num_ftrs = model.classifier[1].in_features
                    model.classifier[1] = nn.Linear(num_ftrs, 2)
                elif model_architecture == "VGG16":
                    num_ftrs = model.classifier[6].in_features
                    model.classifier[6] = nn.Linear(num_ftrs, 2)
    except Exception as e:
        log_callback(f"Error initializing model: {str(e)}")
        log_callback("Defaulting to ResNet18")
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Lists to store metrics for plotting
    train_losses = []
    val_accuracies = []
    
    log_callback(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / train_size
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)
        val_acc = (correct/total)*100
        val_accuracies.append(val_acc)
        
        log_callback(f"Epoch {epoch+1}/{epochs}: Loss {epoch_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if progress_callback:
            progress_callback((epoch+1)/epochs)
    
    # Final validation metrics
    model.eval()
    all_preds = []
    all_labels = []
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Generate classification report
    class_names = ["good", "bad"]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    log_callback("\nClassification Report:")
    log_callback(report)
    
    # Save model
    if not save_folder:
        save_folder = os.getcwd()
    
    # Create a timestamp for the model filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_architecture}_{timestamp}.pth"
    save_path = os.path.join(save_folder, model_filename)
    
    # Save model and metadata
    model_info = {
        "architecture": model_architecture,
        "train_size": train_size,
        "val_size": val_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "final_val_accuracy": val_accuracies[-1],
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "confusion_matrix": cm.tolist(),
        "augmentation_enabled": augmentation_enabled,
        "augmentation_settings": augmentation_settings,
        "timestamp": timestamp
    }
    
    metadata_path = os.path.join(save_folder, f"{model_filename}.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=4)
    except Exception as e:
        log_callback(f"Warning: Could not save model metadata: {str(e)}")
    
    torch.save(model.state_dict(), save_path)
    log_callback(f"Model saved to {save_path}")
    update_recent_list(save_path, "recent_models")
    
    # Return model and training metrics for visualization
    return {
        "model": model,
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "confusion_matrix": cm,
        "class_names": class_names
    }

# ---------------------------
# Prediction Function
# ---------------------------
def predict_image(image_path, model):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return None, None, None
    image_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        max_prob, pred = torch.max(probs, 1)
        confidence = max_prob.item() * 100
        class_names = ["good", "bad"]
        predicted_class = class_names[pred.item()]
    return predicted_class, confidence, img

# ---------------------------
# GUI Using CustomTkinter with Threading and Progress Bar
# ---------------------------
def initialize_app():
    """Initialize the application with settings"""
    # Load settings
    load_settings()
    
    # Set appearance mode and color theme from settings
    ctk.set_appearance_mode(app_settings.get("appearance_mode", "dark"))
    ctk.set_default_color_theme(app_settings.get("color_theme", "blue"))
    
    # Create main window
    app = ctk.CTk()
    app.title(f"{APP_NAME} v{APP_VERSION}")
    app.geometry("1100x780")  # Slightly reduced height
    
    # Create TabView
    tabview = ctk.CTkTabview(app, width=1050, height=720)  # Slightly reduced height
    tabview.pack(padx=20, pady=20, fill="both", expand=True)
    tabview.add("Train")
    tabview.add("Data Visualization")
    tabview.add("Predict")
    tabview.add("Settings")
    tabview.add("UI Help")
    tabview.add("Model Help")
    tabview.set("Train")
    
    try:
        tabview._segmented_button.configure(font=ctk.CTkFont(family="Roboto", size=16))
    except Exception as e:
        print("Could not modify segmented button:", e)
    
    return app, tabview

# ---------------------------
# Thread-Safe Log Callback
# ---------------------------
def log_callback(message):
    app.after(0, update_log, message)

def update_log(message):
    log_text.configure(state="normal")
    log_text.insert("end", f"{message}\n")
    log_text.configure(state="disabled")
    log_text.see("end")

# ---------------------------
# Progress Bar Update Function
# ---------------------------
def update_progress(value):
    progress_bar.set(value)
    progress_text_var.set(f"{int(value*100)}% Completed")

# ---------------------------
# TRAIN TAB WIDGETS
# ---------------------------
def setup_train_tab():
    """Setup the Train tab widgets"""
    train_frame = tabview.tab("Train")
    modern_font = ctk.CTkFont(family="Roboto", size=14)
    
    # Create a title label with larger font
    title_font = ctk.CTkFont(family="Roboto", size=20, weight="bold")
    title_label = ctk.CTkLabel(train_frame, text="Train Your Custom Vision Model", font=title_font)
    title_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 10), sticky="nw")  # Reduced bottom padding
    
    # Create a main container with two columns
    main_container = ctk.CTkFrame(train_frame)
    main_container.grid(row=1, column=0, padx=10, pady=(5, 5), sticky="nsew")  # Reduced padding
    
    # Left column for folder selection
    folder_frame = ctk.CTkFrame(main_container)
    folder_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nw")  # Reduced padding
    
    folder_title = ctk.CTkLabel(folder_frame, text="Data Selection", font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    folder_title.grid(row=0, column=0, columnspan=2, padx=5, pady=3, sticky="nw")  # Reduced padding
    
    good_folder_var = ctk.StringVar()
    bad_folder_var = ctk.StringVar()
    save_folder_var = ctk.StringVar()
    
    def select_good_folder():
        folder = filedialog.askdirectory(title="Select Good Images Folder")
        if folder:
            good_folder_var.set(folder)
            update_recent_list(folder, "recent_good_folders")
            update_folder_stats()
    
    def select_bad_folder():
        folder = filedialog.askdirectory(title="Select Bad Images Folder")
        if folder:
            bad_folder_var.set(folder)
            update_recent_list(folder, "recent_bad_folders")
            update_folder_stats()
    
    def select_save_folder():
        folder = filedialog.askdirectory(title="Select Folder to Save Model")
        if folder:
            save_folder_var.set(folder)
            update_recent_list(folder, "recent_save_folders")
    
    def update_folder_stats():
        """Update statistics about selected folders"""
        good_folder = good_folder_var.get()
        bad_folder = bad_folder_var.get()
        
        good_count = len(load_image_paths(good_folder)) if good_folder else 0
        bad_count = len(load_image_paths(bad_folder)) if bad_folder else 0
        total_count = good_count + bad_count
        
        stats_text = f"Good images: {good_count}\nBad images: {bad_count}\nTotal: {total_count}"
        folder_stats_label.configure(text=stats_text)
    
    # Good folder selection
    good_folder_btn = ctk.CTkButton(folder_frame, text="Select Good Images Folder", command=select_good_folder, font=modern_font, width=200)
    good_folder_btn.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    good_folder_label = ctk.CTkLabel(folder_frame, textvariable=good_folder_var, font=modern_font, wraplength=300)
    good_folder_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
    
    # Recent good folders dropdown
    if app_settings.get("recent_good_folders"):
        recent_good_var = ctk.StringVar(value="Recent Good Folders")
        recent_good_menu = ctk.CTkOptionMenu(
            folder_frame, 
            values=["Recent Good Folders"] + app_settings.get("recent_good_folders", []),
            variable=recent_good_var,
            font=modern_font,
            width=200
        )
        recent_good_menu.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        def on_recent_good_selected(choice):
            if choice != "Recent Good Folders":
                good_folder_var.set(choice)
                update_folder_stats()
        
        recent_good_menu.configure(command=on_recent_good_selected)
    
    # Bad folder selection
    bad_folder_btn = ctk.CTkButton(folder_frame, text="Select Bad Images Folder", command=select_bad_folder, font=modern_font, width=200)
    bad_folder_btn.grid(row=3, column=0, padx=5, pady=5, sticky="w")
    bad_folder_label = ctk.CTkLabel(folder_frame, textvariable=bad_folder_var, font=modern_font, wraplength=300)
    bad_folder_label.grid(row=3, column=1, padx=5, pady=5, sticky="w")
    
    # Recent bad folders dropdown
    if app_settings.get("recent_bad_folders"):
        recent_bad_var = ctk.StringVar(value="Recent Bad Folders")
        recent_bad_menu = ctk.CTkOptionMenu(
            folder_frame, 
            values=["Recent Bad Folders"] + app_settings.get("recent_bad_folders", []),
            variable=recent_bad_var,
            font=modern_font,
            width=200
        )
        recent_bad_menu.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        
        def on_recent_bad_selected(choice):
            if choice != "Recent Bad Folders":
                bad_folder_var.set(choice)
                update_folder_stats()
        
        recent_bad_menu.configure(command=on_recent_bad_selected)
    
    # Save folder selection
    save_folder_btn = ctk.CTkButton(folder_frame, text="Select Model Save Location", command=select_save_folder, font=modern_font, width=200)
    save_folder_btn.grid(row=5, column=0, padx=5, pady=5, sticky="w")
    save_folder_label = ctk.CTkLabel(folder_frame, textvariable=save_folder_var, font=modern_font, wraplength=300)
    save_folder_label.grid(row=5, column=1, padx=5, pady=5, sticky="w")
    
    # Recent save folders dropdown
    if app_settings.get("recent_save_folders"):
        recent_save_var = ctk.StringVar(value="Recent Save Locations")
        recent_save_menu = ctk.CTkOptionMenu(
            folder_frame, 
            values=["Recent Save Locations"] + app_settings.get("recent_save_folders", []),
            variable=recent_save_var,
            font=modern_font,
            width=200
        )
        recent_save_menu.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        
        def on_recent_save_selected(choice):
            if choice != "Recent Save Locations":
                save_folder_var.set(choice)
        
        recent_save_menu.configure(command=on_recent_save_selected)
    
    # Folder statistics
    folder_stats_label = ctk.CTkLabel(folder_frame, text="Good images: 0\nBad images: 0\nTotal: 0", font=modern_font, justify="left")
    folder_stats_label.grid(row=7, column=0, columnspan=2, padx=5, pady=10, sticky="w")
    
    # Right column for training parameters
    params_frame = ctk.CTkFrame(main_container)
    params_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nw")  # Reduced padding
    
    params_title = ctk.CTkLabel(params_frame, text="Training Parameters", font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    params_title.grid(row=0, column=0, columnspan=2, padx=5, pady=3, sticky="nw")  # Reduced padding
    
    # Model architecture selection
    architecture_label = ctk.CTkLabel(params_frame, text="Model Architecture:", font=modern_font)
    architecture_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    
    architecture_var = ctk.StringVar(value=app_settings.get("last_model_architecture", "ResNet18"))
    architecture_menu = ctk.CTkOptionMenu(
        params_frame,
        values=list(MODEL_ARCHITECTURES.keys()),
        variable=architecture_var,
        font=modern_font
    )
    architecture_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")
    
    # Auto adjust parameters
    auto_adjust_var = ctk.BooleanVar(value=app_settings.get("auto_adjust", True))
    auto_adjust_cb = ctk.CTkCheckBox(params_frame, text="Auto Adjust Parameters", variable=auto_adjust_var, font=modern_font)
    auto_adjust_cb.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    
    # Manual batch size and epochs
    manual_batch_size_label = ctk.CTkLabel(params_frame, text="Batch Size:", font=modern_font)
    manual_epochs_label = ctk.CTkLabel(params_frame, text="Epochs:", font=modern_font)
    manual_batch_size_var = ctk.StringVar(value=app_settings.get("default_batch_size", "16"))
    manual_epochs_var = ctk.StringVar(value=app_settings.get("default_epochs", "20"))
    manual_batch_size_entry = ctk.CTkEntry(params_frame, textvariable=manual_batch_size_var, font=modern_font)
    manual_epochs_entry = ctk.CTkEntry(params_frame, textvariable=manual_epochs_var, font=modern_font)
    
    def toggle_manual_entries():
        if auto_adjust_var.get():
            manual_batch_size_label.grid_remove()
            manual_batch_size_entry.grid_remove()
            manual_epochs_label.grid_remove()
            manual_epochs_entry.grid_remove()
        else:
            manual_batch_size_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
            manual_batch_size_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
            manual_epochs_label.grid(row=4, column=0, padx=5, pady=5, sticky="e")
            manual_epochs_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        # Save setting
        app_settings["auto_adjust"] = auto_adjust_var.get()
        save_settings()
    
    auto_adjust_cb.configure(command=toggle_manual_entries)
    toggle_manual_entries()
    
    # Data augmentation frame
    augmentation_frame = ctk.CTkFrame(params_frame)
    augmentation_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=(15, 5), sticky="ew")
    
    augmentation_title = ctk.CTkLabel(augmentation_frame, text="Data Augmentation", font=ctk.CTkFont(family="Roboto", size=14, weight="bold"))
    augmentation_title.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    
    augmentation_enabled_var = ctk.BooleanVar(value=app_settings.get("augmentation_enabled", False))
    augmentation_cb = ctk.CTkCheckBox(augmentation_frame, text="Enable Data Augmentation", variable=augmentation_enabled_var, font=modern_font)
    augmentation_cb.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    
    # Augmentation settings
    aug_settings = app_settings.get("augmentation_settings", {})
    
    # Horizontal flip
    h_flip_var = ctk.BooleanVar(value=aug_settings.get("horizontal_flip", True))
    h_flip_cb = ctk.CTkCheckBox(augmentation_frame, text="Horizontal Flip", variable=h_flip_var, font=modern_font)
    h_flip_cb.grid(row=2, column=0, padx=5, pady=5, sticky="w")
    
    # Vertical flip
    v_flip_var = ctk.BooleanVar(value=aug_settings.get("vertical_flip", False))
    v_flip_cb = ctk.CTkCheckBox(augmentation_frame, text="Vertical Flip", variable=v_flip_var, font=modern_font)
    v_flip_cb.grid(row=2, column=1, padx=5, pady=5, sticky="w")
    
    # Rotation
    rotation_label = ctk.CTkLabel(augmentation_frame, text="Rotation (degrees):", font=modern_font)
    rotation_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
    rotation_var = ctk.DoubleVar(value=float(aug_settings.get("rotation", 15)))
    rotation_slider = ctk.CTkSlider(augmentation_frame, from_=0, to=30, number_of_steps=30, variable=rotation_var)
    rotation_slider.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
    rotation_value = ctk.CTkLabel(augmentation_frame, textvariable=rotation_var, font=modern_font)
    rotation_value.grid(row=3, column=2, padx=5, pady=5, sticky="w")
    
    # Brightness
    brightness_label = ctk.CTkLabel(augmentation_frame, text="Brightness:", font=modern_font)
    brightness_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
    brightness_var = ctk.DoubleVar(value=aug_settings.get("brightness", 0.1))
    brightness_slider = ctk.CTkSlider(augmentation_frame, from_=0, to=0.5, number_of_steps=50, variable=brightness_var)
    brightness_slider.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
    brightness_value = ctk.CTkLabel(augmentation_frame, textvariable=brightness_var, font=modern_font)
    brightness_value.grid(row=4, column=2, padx=5, pady=5, sticky="w")
    
    # Contrast
    contrast_label = ctk.CTkLabel(augmentation_frame, text="Contrast:", font=modern_font)
    contrast_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
    contrast_var = ctk.DoubleVar(value=aug_settings.get("contrast", 0.1))
    contrast_slider = ctk.CTkSlider(augmentation_frame, from_=0, to=0.5, number_of_steps=50, variable=contrast_var)
    contrast_slider.grid(row=5, column=1, padx=5, pady=5, sticky="ew")
    contrast_value = ctk.CTkLabel(augmentation_frame, textvariable=contrast_var, font=modern_font)
    contrast_value.grid(row=5, column=2, padx=5, pady=5, sticky="w")
    
    def save_augmentation_settings():
        """Save augmentation settings to app settings"""
        app_settings["augmentation_enabled"] = augmentation_enabled_var.get()
        app_settings["augmentation_settings"] = {
            "horizontal_flip": h_flip_var.get(),
            "vertical_flip": v_flip_var.get(),
            "rotation": float(rotation_var.get()),
            "brightness": float(brightness_var.get()),
            "contrast": float(contrast_var.get()),
            "saturation": float(contrast_var.get())  # Using contrast as saturation for simplicity
        }
        save_settings()
    
    # Save settings when changed
    augmentation_cb.configure(command=save_augmentation_settings)
    h_flip_cb.configure(command=save_augmentation_settings)
    v_flip_cb.configure(command=save_augmentation_settings)
    rotation_slider.configure(command=lambda _: save_augmentation_settings())
    brightness_slider.configure(command=lambda _: save_augmentation_settings())
    contrast_slider.configure(command=lambda _: save_augmentation_settings())
    
    # Training log and progress
    log_frame = ctk.CTkFrame(train_frame)
    log_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
    
    log_title = ctk.CTkLabel(log_frame, text="Training Log", font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    log_title.pack(anchor="w", padx=5, pady=5)
    
    global log_text, progress_bar, progress_text_var
    log_text = ctk.CTkTextbox(log_frame, width=1000, height=150, font=modern_font)  # Reduced height from 200 to 150
    log_text.pack(padx=10, pady=10, fill="both", expand=True)
    
    progress_text_var = ctk.StringVar(value="0% Completed")
    progress_label = ctk.CTkLabel(log_frame, textvariable=progress_text_var, font=modern_font)
    progress_label.pack(padx=10, pady=(5,0), anchor="w")  # Reduced top padding
    
    progress_bar = ctk.CTkProgressBar(log_frame, width=1000)
    progress_bar.pack(padx=10, pady=5, fill="x")  # Reduced padding
    progress_bar.set(0)
    
    # Results visualization frame
    results_frame = ctk.CTkFrame(train_frame)
    results_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=(5, 5), sticky="nsew")  # Reduced padding
    
    # Start training function
    def start_training():
        # Get folder paths
        good_folder = good_folder_var.get()
        bad_folder = bad_folder_var.get()
        save_folder = save_folder_var.get()
        
        # Validate folders
        if not good_folder or not bad_folder:
            messagebox.showerror("Error", "Please select both Good and Bad image folders.")
            return
        
        # Clear log and reset progress
        log_text.delete("1.0", "end")
        progress_bar.set(0)
        progress_text_var.set("0% Completed")
        
        # Get training parameters
        auto_adjust = auto_adjust_var.get()
        manual_bs = manual_batch_size_var.get() if not auto_adjust else None
        manual_ep = manual_epochs_var.get() if not auto_adjust else None
        model_architecture = architecture_var.get()
        
        # Get augmentation settings
        augmentation_enabled = augmentation_enabled_var.get()
        augmentation_settings = {
            "horizontal_flip": h_flip_var.get(),
            "vertical_flip": v_flip_var.get(),
            "rotation": float(rotation_var.get()),
            "brightness": float(brightness_var.get()),
            "contrast": float(contrast_var.get()),
            "saturation": float(contrast_var.get())  # Using contrast as saturation for simplicity
        }
        
        # Save settings
        app_settings["last_model_architecture"] = model_architecture
        app_settings["default_batch_size"] = manual_bs or "16"
        app_settings["default_epochs"] = manual_ep or "20"
        app_settings["augmentation_enabled"] = augmentation_enabled
        app_settings["augmentation_settings"] = augmentation_settings
        save_settings()
        
        # Log training start
        log_callback("Starting training...")
        log_callback(f"Model architecture: {model_architecture}")
        log_callback(f"Data augmentation: {'Enabled' if augmentation_enabled else 'Disabled'}")
        
        # Start training in a separate thread
        train_thread = threading.Thread(
            target=lambda: train_model(
                good_folder, bad_folder, save_folder, log_callback, auto_adjust, 
                manual_bs, manual_ep, update_progress, model_architecture,
                augmentation_enabled, augmentation_settings
            )
        )
        train_thread.daemon = True  # Allow app to exit even if thread is running
        train_thread.start()
    
    # Training button
    train_btn_frame = ctk.CTkFrame(train_frame)
    train_btn_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=(5, 10), sticky="w")  # Reduced padding
    
    train_btn = ctk.CTkButton(
        train_btn_frame, 
        text="Train Model", 
        command=start_training, 
        font=ctk.CTkFont(family="Roboto", size=16, weight="bold"), 
        width=200,
        height=40,
        fg_color="#28a745",
        hover_color="#218838"
    )
    train_btn.pack(padx=10, pady=5)  # Reduced padding
    
    return {
        "train_frame": train_frame,
        "log_text": log_text,
        "progress_bar": progress_bar,
        "progress_text_var": progress_text_var,
        "results_frame": results_frame
    }

##########################
# SETTINGS TAB WIDGETS
##########################
def setup_settings_tab(tabview):
    """Setup the Settings tab widgets"""
    global appearance_mode_var, color_theme_var, default_batch_size_var, default_epochs_var, default_architecture_var
    
    settings_frame = tabview.tab("Settings")
    modern_font = ctk.CTkFont(family="Roboto", size=14)
    
    # Create a title label with larger font
    title_font = ctk.CTkFont(family="Roboto", size=20, weight="bold")
    settings_title = ctk.CTkLabel(settings_frame, text="Application Settings", font=title_font)
    settings_title.pack(padx=10, pady=(10, 20), anchor="nw")
    
    # Create main container
    settings_container = ctk.CTkFrame(settings_frame)
    settings_container.pack(padx=10, pady=10, fill="both", expand=True)
    
    # Appearance settings
    appearance_frame = ctk.CTkFrame(settings_container)
    appearance_frame.pack(padx=10, pady=10, fill="x")
    
    appearance_title = ctk.CTkLabel(appearance_frame, text="Appearance", font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    appearance_title.pack(padx=5, pady=5, anchor="w")
    
    # Appearance mode
    appearance_mode_label = ctk.CTkLabel(appearance_frame, text="Appearance Mode:", font=modern_font)
    appearance_mode_label.pack(padx=5, pady=5, anchor="w")
    
    appearance_mode_var = ctk.StringVar(value=app_settings.get("appearance_mode", "dark"))
    appearance_mode_menu = ctk.CTkOptionMenu(
        appearance_frame,
        values=["dark", "light", "system"],
        variable=appearance_mode_var,
        font=modern_font,
        width=200
    )
    appearance_mode_menu.pack(padx=20, pady=5, anchor="w")
    
    # Color theme
    color_theme_label = ctk.CTkLabel(appearance_frame, text="Color Theme:", font=modern_font)
    color_theme_label.pack(padx=5, pady=5, anchor="w")
    
    color_theme_var = ctk.StringVar(value=app_settings.get("color_theme", "blue"))
    color_theme_menu = ctk.CTkOptionMenu(
        appearance_frame,
        values=["blue", "green", "dark-blue"],
        variable=color_theme_var,
        font=modern_font,
        width=200
    )
    color_theme_menu.pack(padx=20, pady=5, anchor="w")
    
    # Default settings
    defaults_frame = ctk.CTkFrame(settings_container)
    defaults_frame.pack(padx=10, pady=10, fill="x")
    
    defaults_title = ctk.CTkLabel(defaults_frame, text="Default Training Settings", font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    defaults_title.pack(padx=5, pady=5, anchor="w")
    
    # Default batch size
    default_batch_size_label = ctk.CTkLabel(defaults_frame, text="Default Batch Size:", font=modern_font)
    default_batch_size_label.pack(padx=5, pady=5, anchor="w")
    
    default_batch_size_var = ctk.StringVar(value=app_settings.get("default_batch_size", "16"))
    default_batch_size_entry = ctk.CTkEntry(defaults_frame, textvariable=default_batch_size_var, font=modern_font, width=200)
    default_batch_size_entry.pack(padx=20, pady=5, anchor="w")
    
    # Default epochs
    default_epochs_label = ctk.CTkLabel(defaults_frame, text="Default Epochs:", font=modern_font)
    default_epochs_label.pack(padx=5, pady=5, anchor="w")
    
    default_epochs_var = ctk.StringVar(value=app_settings.get("default_epochs", "20"))
    default_epochs_entry = ctk.CTkEntry(defaults_frame, textvariable=default_epochs_var, font=modern_font, width=200)
    default_epochs_entry.pack(padx=20, pady=5, anchor="w")
    
    # Default model architecture
    default_architecture_label = ctk.CTkLabel(defaults_frame, text="Default Model Architecture:", font=modern_font)
    default_architecture_label.pack(padx=5, pady=5, anchor="w")
    
    default_architecture_var = ctk.StringVar(value=app_settings.get("last_model_architecture", "ResNet18"))
    default_architecture_menu = ctk.CTkOptionMenu(
        defaults_frame,
        values=list(MODEL_ARCHITECTURES.keys()),
        variable=default_architecture_var,
        font=modern_font,
        width=200
    )
    default_architecture_menu.pack(padx=20, pady=5, anchor="w")
    
    # Recent files management
    recent_files_frame = ctk.CTkFrame(settings_container)
    recent_files_frame.pack(padx=10, pady=10, fill="x")
    
    recent_files_title = ctk.CTkLabel(recent_files_frame, text="Recent Files Management", font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    recent_files_title.pack(padx=5, pady=5, anchor="w")
    
    # Clear recent files
    clear_recent_btn = ctk.CTkButton(
        recent_files_frame,
        text="Clear All Recent Files",
        command=lambda: clear_recent_files(),
        font=modern_font,
        width=200,
        fg_color="#dc3545",
        hover_color="#c82333"
    )
    clear_recent_btn.pack(padx=20, pady=10, anchor="w")
    
    # Save settings button
    save_settings_btn = ctk.CTkButton(
        settings_container,
        text="Save Settings",
        command=lambda: save_settings_from_ui(),
        font=ctk.CTkFont(family="Roboto", size=16, weight="bold"),
        width=200,
        height=40,
        fg_color="#28a745",
        hover_color="#218838"
    )
    save_settings_btn.pack(padx=10, pady=20, anchor="center")
    
    return settings_frame

##########################
# DATA VISUALIZATION TAB WIDGETS
##########################
def setup_data_visualization_tab(tabview):
    """Setup the Data Visualization tab widgets"""
    viz_frame = tabview.tab("Data Visualization")
    modern_font = ctk.CTkFont(family="Roboto", size=14)
    
    # Create a title label with larger font
    title_font = ctk.CTkFont(family="Roboto", size=20, weight="bold")
    title_label = ctk.CTkLabel(viz_frame, text="Data Visualization", font=title_font)
    title_label.pack(padx=10, pady=(10, 20), anchor="nw")
    
    # Create main container
    main_container = ctk.CTkFrame(viz_frame)
    main_container.pack(padx=10, pady=10, fill="both", expand=True)
    
    # Instructions
    instructions = ctk.CTkLabel(
        main_container, 
        text="This tab will display visualizations of your dataset after training.\n"
             "Train a model first to see visualizations here.",
        font=modern_font,
        justify="left"
    )
    instructions.pack(padx=20, pady=20, anchor="nw")
    
    # Placeholder for visualizations
    viz_placeholder = ctk.CTkFrame(main_container)
    viz_placeholder.pack(padx=20, pady=20, fill="both", expand=True)
    
    placeholder_label = ctk.CTkLabel(
        viz_placeholder,
        text="Train a model to see visualizations",
        font=modern_font
    )
    placeholder_label.pack(padx=20, pady=20)
    
    return viz_frame

##########################
# PREDICT TAB WIDGETS
##########################
def setup_predict_tab(tabview):
    """Setup the Predict tab widgets"""
    predict_frame = tabview.tab("Predict")
    modern_font = ctk.CTkFont(family="Roboto", size=14)
    
    # Create a title label with larger font
    title_font = ctk.CTkFont(family="Roboto", size=20, weight="bold")
    title_label = ctk.CTkLabel(predict_frame, text="Predict with Your Model", font=title_font)
    title_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 20), sticky="nw")
    
    # Create a main container with two columns
    main_container = ctk.CTkFrame(predict_frame)
    main_container.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
    
    # Left column for model selection and image upload
    left_frame = ctk.CTkFrame(main_container)
    left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
    
    # Model selection
    model_frame = ctk.CTkFrame(left_frame)
    model_frame.pack(padx=10, pady=10, fill="x")
    
    model_title = ctk.CTkLabel(model_frame, text="Model Selection", font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    model_title.pack(padx=5, pady=5, anchor="w")
    
    model_path_var = ctk.StringVar()
    model_path_entry = ctk.CTkEntry(model_frame, textvariable=model_path_var, width=300, font=modern_font)
    model_path_entry.pack(padx=5, pady=5, fill="x")
    
    model_btn_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
    model_btn_frame.pack(padx=5, pady=5, fill="x")
    
    def select_model():
        model_file = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        if model_file:
            model_path_var.set(model_file)
            update_recent_list(model_file, "recent_models")
    
    select_model_btn = ctk.CTkButton(
        model_btn_frame,
        text="Select Model",
        command=select_model,
        font=modern_font,
        width=150
    )
    select_model_btn.pack(side="left", padx=5, pady=5)
    
    # Recent models dropdown
    recent_models_var = ctk.StringVar(value="Recent Models")
    
    def on_recent_model_selected(choice):
        if choice != "Recent Models" and Path(choice).exists():
            model_path_var.set(choice)
    
    if app_settings.get("recent_models"):
        recent_models_menu = ctk.CTkOptionMenu(
            model_btn_frame,
            values=["Recent Models"] + app_settings.get("recent_models", []),
            command=on_recent_model_selected,
            variable=recent_models_var,
            font=modern_font,
            width=150
        )
        recent_models_menu.pack(side="left", padx=5, pady=5)
    
    # Load model button
    def load_model_for_prediction():
        global model
        model_path = model_path_var.get()
        
        if not model_path or not Path(model_path).exists():
            messagebox.showerror("Error", "Please select a valid model file.")
            return
        
        try:
            log_callback(f"Loading model from {model_path}...")
            
            # Get the architecture from the filename or use default
            architecture_name = "ResNet18"  # Default
            for arch in MODEL_ARCHITECTURES.keys():
                if arch.lower() in Path(model_path).name.lower():
                    architecture_name = arch
                    break
            
            # Create model with the same architecture
            model = MODEL_ARCHITECTURES[architecture_name](weights=None)
            
            # Modify the final layer to match our binary classification
            if hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, 2)
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, 2)
                else:
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, 2)
            
            # Load the model weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            log_callback(f"Model loaded successfully! Architecture: {architecture_name}")
            messagebox.showinfo("Success", "Model loaded successfully!")
            
            # Enable prediction buttons
            predict_image_btn.configure(state="normal")
            start_video_btn.configure(state="normal")
            
        except Exception as e:
            log_callback(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    load_model_btn = ctk.CTkButton(
        model_frame,
        text="Load Model",
        command=load_model_for_prediction,
        font=ctk.CTkFont(family="Roboto", size=16, weight="bold"),
        width=200,
        height=40,
        fg_color="#007bff",
        hover_color="#0069d9"
    )
    load_model_btn.pack(padx=10, pady=10)
    
    # Image prediction
    image_frame = ctk.CTkFrame(left_frame)
    image_frame.pack(padx=10, pady=10, fill="x")
    
    image_title = ctk.CTkLabel(image_frame, text="Image Prediction", font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    image_title.pack(padx=5, pady=5, anchor="w")
    
    image_path_var = ctk.StringVar()
    image_path_entry = ctk.CTkEntry(image_frame, textvariable=image_path_var, width=300, font=modern_font)
    image_path_entry.pack(padx=5, pady=5, fill="x")
    
    def select_image():
        image_file = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        if image_file:
            image_path_var.set(image_file)
    
    select_image_btn = ctk.CTkButton(
        image_frame,
        text="Select Image",
        command=select_image,
        font=modern_font,
        width=200
    )
    select_image_btn.pack(padx=5, pady=5)
    
    # Right column for results and visualization
    right_frame = ctk.CTkFrame(main_container)
    right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    
    result_title = ctk.CTkLabel(right_frame, text="Prediction Results", font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    result_title.pack(padx=5, pady=5, anchor="w")
    
    # Image display
    image_display = ctk.CTkLabel(right_frame, text="Image will be displayed here", font=modern_font)
    image_display.pack(padx=10, pady=10, fill="both", expand=True)
    
    # Prediction result
    result_var = ctk.StringVar(value="Prediction: N/A")
    result_label = ctk.CTkLabel(right_frame, textvariable=result_var, font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    result_label.pack(padx=10, pady=10)
    
    confidence_var = ctk.StringVar(value="Confidence: N/A")
    confidence_label = ctk.CTkLabel(right_frame, textvariable=confidence_var, font=modern_font)
    confidence_label.pack(padx=10, pady=5)
    
    # Video frame
    video_frame = ctk.CTkFrame(predict_frame)
    video_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
    
    video_title = ctk.CTkLabel(video_frame, text="Live Video Prediction", font=ctk.CTkFont(family="Roboto", size=16, weight="bold"))
    video_title.pack(padx=5, pady=5, anchor="w")
    
    # Video source selection
    video_source_frame = ctk.CTkFrame(video_frame, fg_color="transparent")
    video_source_frame.pack(padx=10, pady=5, fill="x")
    
    video_source_var = ctk.StringVar(value="Webcam")
    webcam_radio = ctk.CTkRadioButton(
        video_source_frame,
        text="Webcam",
        variable=video_source_var,
        value="Webcam",
        font=modern_font
    )
    webcam_radio.pack(side="left", padx=10, pady=5)
    
    video_file_radio = ctk.CTkRadioButton(
        video_source_frame,
        text="Video File",
        variable=video_source_var,
        value="Video File",
        font=modern_font
    )
    video_file_radio.pack(side="left", padx=10, pady=5)
    
    # Video file selection
    video_path_var = ctk.StringVar()
    video_path_frame = ctk.CTkFrame(video_frame, fg_color="transparent")
    video_path_frame.pack(padx=10, pady=5, fill="x")
    
    video_path_entry = ctk.CTkEntry(video_path_frame, textvariable=video_path_var, width=400, font=modern_font)
    video_path_entry.pack(side="left", padx=5, pady=5, fill="x", expand=True)
    
    def select_video_file():
        video_file = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if video_file:
            video_path_var.set(video_file)
            video_source_var.set("Video File")
    
    select_video_btn = ctk.CTkButton(
        video_path_frame,
        text="Browse",
        command=select_video_file,
        font=modern_font,
        width=100
    )
    select_video_btn.pack(side="left", padx=5, pady=5)
    
    # Video display
    video_display = ctk.CTkLabel(video_frame, text="Video will be displayed here", font=modern_font)
    video_display.pack(padx=10, pady=10, fill="both", expand=True)
    
    # Video controls
    video_controls = ctk.CTkFrame(video_frame, fg_color="transparent")
    video_controls.pack(padx=10, pady=10)
    
    # Global variables for video
    video_running = False
    cap = None
    
    def start_video():
        global video_running, cap
        
        if model is None:
            messagebox.showerror("Error", "Please load a model first.")
            return
        
        try:
            # Check video source
            if video_source_var.get() == "Webcam":
                cap = cv2.VideoCapture(0)
            else:
                video_path = video_path_var.get()
                if not video_path or not Path(video_path).exists():
                    messagebox.showerror("Error", "Please select a valid video file.")
                    return
                cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video source.")
                return
            
            video_running = True
            start_video_btn.configure(state="disabled")
            stop_video_btn.configure(state="normal")
            
            update_video_feed()
        except Exception as e:
            messagebox.showerror("Error", f"Error starting video: {str(e)}")
    
    def stop_video():
        global video_running, cap
        
        video_running = False
        if cap is not None:
            cap.release()
        
        start_video_btn.configure(state="normal")
        stop_video_btn.configure(state="disabled")
        
        # Reset video display
        video_display.configure(image=None, text="Video stopped")
    
    def update_video_feed():
        global video_running, cap
        
        if not video_running:
            return
        
        ret, frame = cap.read()
        if not ret:
            # If video file ended, stop playback
            if video_source_var.get() == "Video File":
                stop_video()
                messagebox.showinfo("Video Playback", "Video playback completed.")
                return
            else:
                stop_video()
                messagebox.showerror("Error", "Failed to capture frame from webcam.")
                return
        
        # Process frame for prediction
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize for display
        display_img = img_pil.copy()
        display_img.thumbnail((640, 480))
        
        # Prepare for model
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            confidence_val = confidence.item()
            predicted_class = "good" if predicted.item() == 0 else "bad"
        
        # Add prediction text to frame
        result_text = f"Prediction: {predicted_class.upper()}"
        confidence_text = f"Confidence: {confidence_val:.2f}"
        
        # Convert to CTk image
        img_tk = ImageTk.PhotoImage(display_img)
        
        # Update display
        video_display.configure(image=img_tk, text="")
        video_display.image = img_tk  # Keep a reference
        
        # Update result text
        result_var.set(result_text)
        confidence_var.set(confidence_text)
        
        # Add Grad-CAM visualization if confidence is high enough
        if confidence_val > 0.7:
            try:
                # Get the last convolutional layer for Grad-CAM
                if hasattr(model, 'layer4'):
                    target_layer = model.layer4[-1]
                elif hasattr(model, 'features'):
                    # For models like MobileNet, VGG
                    target_layer = model.features[-1]
                else:
                    target_layer = None
                
                if target_layer is not None:
                    # Generate Grad-CAM
                    heatmap = generate_gradcam(model, img_tensor, target_layer)
                    
                    # Resize heatmap to match frame size
                    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
                    
                    # Apply Grad-CAM overlay with bounding boxes
                    overlay_gradcam_boxes(frame, heatmap_resized, threshold=0.5)
                    
                    # Convert back to display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    img_pil.thumbnail((640, 480))
                    img_tk = ImageTk.PhotoImage(img_pil)
                    
                    # Update display with Grad-CAM
                    video_display.configure(image=img_tk)
                    video_display.image = img_tk
            except Exception as e:
                print(f"Error applying Grad-CAM: {str(e)}")
        
        # Schedule the next update
        predict_frame.after(33, update_video_feed)  # ~30 FPS
    
    def predict_single_image():
        if model is None:
            messagebox.showerror("Error", "Please load a model first.")
            return
        
        image_path = image_path_var.get()
        if not image_path or not Path(image_path).exists():
            messagebox.showerror("Error", "Please select a valid image file.")
            return
        
        try:
            # Load and preprocess the image
            img = Image.open(image_path).convert('RGB')
            
            # Resize for display
            display_img = img.copy()
            display_img.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(display_img)
            
            # Update image display
            image_display.configure(image=img_tk, text="")
            image_display.image = img_tk  # Keep a reference
            
            # Predict
            predicted_class, confidence_val, _ = predict_image(image_path, model)
            
            # Update result
            result_var.set(f"Prediction: {predicted_class.upper()}")
            confidence_var.set(f"Confidence: {confidence_val:.2f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error predicting image: {str(e)}")
    
    # Prediction buttons
    predict_image_btn = ctk.CTkButton(
        image_frame,
        text="Predict Image",
        command=predict_single_image,
        font=ctk.CTkFont(family="Roboto", size=16, weight="bold"),
        width=200,
        height=40,
        fg_color="#28a745",
        hover_color="#218838",
        state="disabled"  # Initially disabled until model is loaded
    )
    predict_image_btn.pack(padx=10, pady=10)
    
    # Video control buttons
    start_video_btn = ctk.CTkButton(
        video_controls,
        text="Start Live Video",
        command=start_video,
        font=modern_font,
        width=200,
        fg_color="#28a745",
        hover_color="#218838",
        state="disabled"  # Initially disabled until model is loaded
    )
    start_video_btn.pack(side="left", padx=10, pady=10)
    
    stop_video_btn = ctk.CTkButton(
        video_controls,
        text="Stop Video",
        command=stop_video,
        font=modern_font,
        width=200,
        fg_color="#dc3545",
        hover_color="#c82333",
        state="disabled"  # Initially disabled until video is started
    )
    stop_video_btn.pack(side="left", padx=10, pady=10)
    
    return predict_frame

##########################
# UI HELP TAB WIDGETS
##########################
def setup_ui_help_tab(tabview):
    """Setup the UI Help tab widgets"""
    help_frame = tabview.tab("UI Help")
    modern_font = ctk.CTkFont(family="Roboto", size=14)
    
    # Create a title label with larger font
    title_font = ctk.CTkFont(family="Roboto", size=20, weight="bold")
    title_label = ctk.CTkLabel(help_frame, text="UI Help & Instructions", font=title_font)
    title_label.pack(padx=10, pady=(10, 20), anchor="nw")
    
    # Create scrollable frame for help content
    help_scroll = ctk.CTkScrollableFrame(help_frame)
    help_scroll.pack(padx=10, pady=10, fill="both", expand=True)
    
    # Help sections
    sections = [
        {
            "title": "Getting Started",
            "content": (
                "Welcome to AI Vision Trainer! This application helps you train a custom vision model "
                "to classify images into 'good' and 'bad' categories.\n\n"
                "To get started, follow these steps:\n"
                "1. Select folders containing your 'good' and 'bad' images\n"
                "2. Choose where to save your trained model\n"
                "3. Configure training parameters (or use auto-adjust)\n"
                "4. Click 'Train Model' to begin training"
            )
        },
        {
            "title": "Train Tab",
            "content": (
                "The Train tab is where you'll prepare and train your model:\n\n"
                "• Data Selection: Choose folders for 'good' and 'bad' images, and where to save the model\n"
                "• Training Parameters: Configure batch size, epochs, and model architecture\n"
                "• Data Augmentation: Enable and configure augmentation to improve model performance\n"
                "• Training Log: View real-time progress and results during training\n"
                "• Results: After training, view performance metrics and visualizations"
            )
        },
        {
            "title": "Data Visualization Tab",
            "content": (
                "After training, the Data Visualization tab shows detailed performance metrics:\n\n"
                "• Training Loss: How the model's error decreased during training\n"
                "• Validation Accuracy: How accurately the model classifies new images\n"
                "• Confusion Matrix: Detailed breakdown of correct and incorrect predictions\n"
                "• Summary Statistics: Key performance metrics in an easy-to-read format\n"
                "• Export: Save all results to CSV files for further analysis"
            )
        },
        {
            "title": "Predict Tab",
            "content": (
                "Use the Predict tab to test your trained model on new images:\n\n"
                "• Load Model: Select a previously trained model file (.pth)\n"
                "• Single Image Prediction: Upload and classify individual images\n"
                "• Live Video: Use your webcam for real-time classification\n"
                "• Grad-CAM Visualization: See which parts of the image influenced the model's decision\n"
                "• Confidence Score: View how confident the model is in its prediction"
            )
        },
        {
            "title": "Settings Tab",
            "content": (
                "Customize the application in the Settings tab:\n\n"
                "• Appearance: Change the theme and color scheme\n"
                "• Default Values: Set default batch size, epochs, and model architecture\n"
                "• Recent Files: Clear history of recently used folders and models"
            )
        },
        {
            "title": "Tips & Tricks",
            "content": (
                "For best results with your vision model:\n\n"
                "• Use at least 50-100 images per category for basic training\n"
                "• Ensure images are representative of real-world conditions\n"
                "• Enable data augmentation for smaller datasets\n"
                "• Try different model architectures for your specific use case\n"
                "• Use Grad-CAM to understand what features the model is focusing on\n"
                "• Save multiple versions of your model to compare performance"
            )
        }
    ]
    
    # Add each section to the scrollable frame
    for i, section in enumerate(sections):
        # Section title
        section_title = ctk.CTkLabel(
            help_scroll, 
            text=section["title"], 
            font=ctk.CTkFont(family="Roboto", size=16, weight="bold")
        )
        section_title.pack(padx=10, pady=(20 if i > 0 else 10, 5), anchor="w")
        
        # Section content
        section_content = ctk.CTkLabel(
            help_scroll,
            text=section["content"],
            font=modern_font,
            justify="left",
            wraplength=800
        )
        section_content.pack(padx=20, pady=5, anchor="w")
    
    return help_frame

##########################
# MODEL HELP TAB WIDGETS
##########################
def setup_model_help_tab(tabview):
    """Setup the Model Help tab widgets"""
    help_frame = tabview.tab("Model Help")
    modern_font = ctk.CTkFont(family="Roboto", size=14)
    
    # Create a title label with larger font
    title_font = ctk.CTkFont(family="Roboto", size=20, weight="bold")
    title_label = ctk.CTkLabel(help_frame, text="Model Information & Help", font=title_font)
    title_label.pack(padx=10, pady=(10, 20), anchor="nw")
    
    # Create scrollable frame for help content
    help_scroll = ctk.CTkScrollableFrame(help_frame)
    help_scroll.pack(padx=10, pady=10, fill="both", expand=True)
    
    # Help sections
    sections = [
        {
            "title": "Available Model Architectures",
            "content": (
                "This application offers several pre-trained model architectures:\n\n"
                "• ResNet18: A lightweight model good for beginners, fast training\n"
                "• ResNet34: Medium-sized model with improved accuracy\n"
                "• ResNet50: Larger model with higher potential accuracy, requires more data\n"
                "• MobileNetV2: Optimized for mobile/edge devices, very efficient\n"
                "• EfficientNetB0: Excellent balance of accuracy and efficiency\n"
                "• VGG16: Classic architecture, larger but sometimes more robust\n\n"
                "Start with ResNet18 for quick experiments, then try others for better performance."
            )
        },
        {
            "title": "Training Parameters Explained",
            "content": (
                "Understanding key training parameters:\n\n"
                "• Batch Size: Number of images processed together. Larger batches use more memory but can train faster.\n"
                "• Epochs: Complete passes through the dataset. More epochs allow more learning but risk overfitting.\n"
                "• Learning Rate: Controls how quickly the model adapts. Auto-adjusted in this application.\n"
                "• Auto-Adjust: Automatically sets batch size and epochs based on your dataset size.\n"
                "• Data Augmentation: Creates variations of your images to improve model generalization."
            )
        },
        {
            "title": "Data Augmentation Options",
            "content": (
                "Data augmentation creates variations of your training images:\n\n"
                "• Horizontal Flip: Mirrors images horizontally\n"
                "• Vertical Flip: Mirrors images vertically\n"
                "• Rotation: Rotates images by random angles up to the specified degree\n"
                "• Brightness: Adjusts image brightness randomly within the specified range\n"
                "• Contrast: Adjusts image contrast randomly within the specified range\n"
                "• Saturation: Adjusts color saturation randomly within the specified range\n\n"
                "Enable augmentation when you have limited training data or to improve model robustness."
            )
        },
        {
            "title": "Understanding Model Performance",
            "content": (
                "After training, evaluate your model using these metrics:\n\n"
                "• Training Loss: Should decrease over time. Plateauing indicates learning has stopped.\n"
                "• Validation Accuracy: Percentage of correctly classified validation images.\n"
                "• Confusion Matrix: Detailed breakdown of predictions:\n"
                "  - True Positives: 'Good' images correctly classified as 'good'\n"
                "  - False Positives: 'Bad' images incorrectly classified as 'good'\n"
                "  - False Negatives: 'Good' images incorrectly classified as 'bad'\n"
                "  - True Negatives: 'Bad' images correctly classified as 'bad'\n\n"
                "Aim for high accuracy and balanced performance across both classes."
            )
        },
        {
            "title": "Grad-CAM Visualization",
            "content": (
                "Gradient-weighted Class Activation Mapping (Grad-CAM) helps visualize what the model is 'looking at':\n\n"
                "• Heat maps highlight regions that influenced the model's decision\n"
                "• Red boxes show areas of highest activation\n"
                "• Use this to verify the model is focusing on relevant features\n"
                "• If the model focuses on irrelevant areas, you may need more or better training data\n\n"
                "Grad-CAM is available in the Predict tab when using live video with high-confidence predictions."
            )
        },
        {
            "title": "Tips for Better Models",
            "content": (
                "Advanced tips for improving model performance:\n\n"
                "• Dataset Quality: Ensure images are clear and representative of real-world conditions\n"
                "• Dataset Size: More images generally lead to better performance (aim for 100+ per class)\n"
                "• Class Balance: Try to have a similar number of 'good' and 'bad' images\n"
                "• Image Variety: Include different angles, lighting conditions, and backgrounds\n"
                "• Transfer Learning: This app uses pre-trained models that already understand basic image features\n"
                "• Fine-Tuning: Try different architectures and training parameters for your specific use case\n"
                "• Validation: Always check model performance on new images it hasn't seen during training"
            )
        },
        {
            "title": "Additional Resources",
            "content": (
                "To learn more about computer vision and deep learning:\n\n"
                "• PyTorch Documentation: https://pytorch.org/docs/stable/index.html\n"
                "• TorchVision Models: https://pytorch.org/vision/stable/models.html\n"
                "• Deep Learning Basics: https://www.deeplearningbook.org/\n"
                "• Computer Vision Tutorials: https://www.pyimagesearch.com/\n"
                "• Data Augmentation Guide: https://pytorch.org/vision/stable/transforms.html"
            )
        }
    ]
    
    # Add each section to the scrollable frame
    for i, section in enumerate(sections):
        # Section title
        section_title = ctk.CTkLabel(
            help_scroll, 
            text=section["title"], 
            font=ctk.CTkFont(family="Roboto", size=16, weight="bold")
        )
        section_title.pack(padx=10, pady=(20 if i > 0 else 10, 5), anchor="w")
        
        # Section content
        section_content = ctk.CTkLabel(
            help_scroll,
            text=section["content"],
            font=modern_font,
            justify="left",
            wraplength=800
        )
        section_content.pack(padx=20, pady=5, anchor="w")
        
        # Add hyperlink functionality for the resources section
        if section["title"] == "Additional Resources":
            links_frame = ctk.CTkFrame(help_scroll, fg_color="transparent")
            links_frame.pack(padx=20, pady=10, anchor="w")
            
            def open_url(url):
                webbrowser.open_new_tab(url)
            
            resources = [
                ("PyTorch Documentation", "https://pytorch.org/docs/stable/index.html"),
                ("TorchVision Models", "https://pytorch.org/vision/stable/models.html"),
                ("Deep Learning Book", "https://www.deeplearningbook.org/"),
                ("PyImageSearch Tutorials", "https://www.pyimagesearch.com/"),
                ("TorchVision Transforms", "https://pytorch.org/vision/stable/transforms.html")
            ]
            
            for name, url in resources:
                link_btn = ctk.CTkButton(
                    links_frame,
                    text=name,
                    command=lambda u=url: open_url(u),
                    font=modern_font,
                    fg_color="#007bff",
                    hover_color="#0056b3",
                    width=200
                )
                link_btn.pack(padx=5, pady=5, anchor="w")
    
    return help_frame

# ---------------------------
# Run the Application
# ---------------------------
def run_application():
    """Initialize and run the application"""
    global app, tabview, log_text, progress_bar, progress_text_var, results_frame
    
    # Initialize the app and tabview
    app, tabview = initialize_app()
    
    # Setup tabs
    train_tab_widgets = setup_train_tab()
    setup_data_visualization_tab(tabview)
    setup_predict_tab(tabview)
    setup_settings_tab(tabview)
    setup_ui_help_tab(tabview)
    setup_model_help_tab(tabview)
    
    # Get references to important widgets
    log_text = train_tab_widgets["log_text"]
    progress_bar = train_tab_widgets["progress_bar"]
    progress_text_var = train_tab_widgets["progress_text_var"]
    results_frame = train_tab_widgets["results_frame"]
    
    # Initialize welcome message
    log_callback("Welcome to the AI Vision Trainer!")
    log_callback("Select Good & Bad image Folders")
    log_callback("Select Trained Model Saved Output Location")
    log_callback("Awaiting User Input...")
    
    # Start the main loop
    app.mainloop()

# Run the application
if __name__ == "__main__":
    run_application()

def train_and_visualize(good_folder, bad_folder, save_folder, log_callback, auto_adjust, 
                       manual_bs, manual_ep, progress_callback, model_architecture,
                       augmentation_enabled, augmentation_settings):
    """Train the model and visualize results"""
    global results_frame
    
    try:
        # Train model
        result = train_model(
            good_folder, bad_folder, save_folder, log_callback, auto_adjust, 
            manual_bs, manual_ep, progress_callback, model_architecture,
            augmentation_enabled, augmentation_settings
        )
        
        # Visualize results
        if result and "train_losses" in result and "val_accuracies" in result and "confusion_matrix" in result:
            # Visualize the model performance
            visualize_model_performance(
                result["train_losses"], 
                result["val_accuracies"], 
                result["confusion_matrix"],
                result["class_names"],
                results_frame
            )
            
            log_callback("Training complete! Results visualized.")
        else:
            log_callback("Training failed or incomplete results returned.")
    except Exception as e:
        log_callback(f"Error in training and visualization: {str(e)}")
        traceback.print_exc()

# Functions for settings
def save_settings_from_ui():
    """Save settings from UI to app_settings and file"""
    global appearance_mode_var, color_theme_var, default_batch_size_var, default_epochs_var, default_architecture_var
    
    # Update app_settings from UI
    app_settings["appearance_mode"] = appearance_mode_var.get()
    app_settings["color_theme"] = color_theme_var.get()
    app_settings["default_batch_size"] = default_batch_size_var.get()
    app_settings["default_epochs"] = default_epochs_var.get()
    app_settings["last_model_architecture"] = default_architecture_var.get()
    
    # Save to file
    save_settings()
    
    # Apply appearance settings
    ctk.set_appearance_mode(app_settings["appearance_mode"])
    ctk.set_default_color_theme(app_settings["color_theme"])
    
    messagebox.showinfo("Settings Saved", "Your settings have been saved and applied.")

def clear_recent_files():
    """Clear all recent files lists"""
    if messagebox.askyesno("Clear Recent Files", "Are you sure you want to clear all recent files history?"):
        app_settings["recent_good_folders"] = []
        app_settings["recent_bad_folders"] = []
        app_settings["recent_save_folders"] = []
        app_settings["recent_models"] = []
        save_settings()
        messagebox.showinfo("Recent Files Cleared", "All recent files history has been cleared.")

# ---------------------------
# Model Performance Visualization
# ---------------------------
def visualize_model_performance(train_losses, val_accuracies, confusion_matrix, class_names, parent_widget):
    """Create visualizations for model performance metrics"""
    # Clear any existing widgets
    for widget in parent_widget.winfo_children():
        widget.destroy()
    
    # Create a frame for the plots
    plots_frame = ctk.CTkFrame(parent_widget)
    plots_frame.pack(padx=10, pady=10, fill="both", expand=True)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(10, 8))
    
    # Training loss plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(train_losses, 'b-')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Validation accuracy plot
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(val_accuracies, 'g-')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    # Confusion matrix plot
    ax3 = fig.add_subplot(2, 2, 3)
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # Classification report
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    # Generate dummy y_true and y_pred from confusion matrix for demonstration
    # In a real scenario, you would use actual labels and predictions
    y_true = []
    y_pred = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            y_true.extend([i] * confusion_matrix[i, j])
            y_pred.extend([j] * confusion_matrix[i, j])
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    ax4.text(0, 0.5, report, fontsize=10, family='monospace')
    
    # Adjust layout
    plt.tight_layout()
    
    # Embed the plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plots_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def update_progress(value):
    progress_bar.set(value)
    progress_text_var.set(f"{int(value*100)}% Completed")

# ---------------------------
# GUI Tab Setup
# ---------------------------