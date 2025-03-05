import os
import time
import threading
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageGrab, ImageTk
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

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
# Custom Dataset Definition
# ---------------------------
class CustomImageDataset(Dataset):
    def __init__(self, good_paths, bad_paths, transform=None):
        # Label 0 for "good", 1 for "bad"
        self.image_paths = good_paths + bad_paths
        self.labels = [0]*len(good_paths) + [1]*len(bad_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def load_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(ALLOWED_EXTS)]

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


from PIL import ImageTk

# ---------------------------
# New Dataset for Manual ROI Images
# ---------------------------
class ManualROIDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        data_list: list of tuples (PIL image, label)
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img, label = self.data_list[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def select_roi_for_image(img):
    # Get maximum allowed dimensions (80% of screen width and height)
    max_width = app.winfo_screenwidth() * 0.8
    max_height = app.winfo_screenheight() * 0.8
    orig_width, orig_height = img.size
    scale = min(1.0, max_width / orig_width, max_height / orig_height)
    if scale < 1.0:
        new_size = (int(orig_width * scale), int(orig_height * scale))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    else:
        img_resized = img

    # Create a blocking Toplevel window for ROI selection using the resized image
    roi_win = tk.Toplevel(app)
    roi_win.title("Select ROI")
    tk_img = ImageTk.PhotoImage(img_resized)
    canvas = tk.Canvas(roi_win, width=img_resized.width, height=img_resized.height, cursor="cross")
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    start_x, start_y = None, None
    rect = None
    result = [None]  # mutable container

    def on_button_press(event):
        nonlocal start_x, start_y, rect
        start_x, start_y = event.x, event.y
        rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="red", width=2)

    def on_move(event):
        canvas.coords(rect, start_x, start_y, event.x, event.y)

    def on_button_release(event):
        nonlocal result
        end_x, end_y = event.x, event.y
        roi_win.destroy()
        bbox = (min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y))
        # Map the coordinates back to the original image size if scaling was applied.
        if scale < 1.0:
            inv_scale = 1.0 / scale
            bbox = (int(bbox[0] * inv_scale), int(bbox[1] * inv_scale),
                    int(bbox[2] * inv_scale), int(bbox[3] * inv_scale))
        cropped = img.crop(bbox)
        result[0] = cropped

    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_button_release)
    roi_win.wait_window()  # block until window is closed
    return result[0]

# ---------------------------
# Modify the Training Function to optionally accept a pre-built dataset
# ---------------------------
def train_model(good_folder, bad_folder, save_folder, log_callback,
                auto_adjust=True, manual_batch_size=None, manual_epochs=None,
                progress_callback=None, manual_dataset=None):
    if manual_dataset is not None:
        dataset = manual_dataset
    else:
        good_paths = load_image_paths(good_folder)
        bad_paths = load_image_paths(bad_folder)
        if len(good_paths) == 0 or len(bad_paths) == 0:
            log_callback("Error: One of the folders has no images.")
            return None
        dataset = CustomImageDataset(good_paths, bad_paths, transform=transform)
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
            log_callback("Error: Manual batch size and epochs must be integers.")
            return None
    
    start_time = time.time()
    log_callback(f"Training started at: {time.ctime(start_time)}")
    log_callback(f"Training device: {device}")
    log_callback(f"Total images: {total_images} (Train: {train_size}, Val: {val_size}), Batch size: {batch_size}")
    
    global model
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    log_callback(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        if cancel_training.is_set():
            log_callback("Training cancelled.")
            return
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
        log_callback(f"Epoch {epoch+1}/{epochs}: Loss {epoch_loss:.4f}")
        if progress_callback:
            progress_callback((epoch+1)/epochs)
    
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
    log_callback(f"Validation Accuracy: {val_acc:.2f}%")
    
    if not save_folder:
        save_folder = os.getcwd()
    save_path = os.path.join(save_folder, "user_trained_model.pth")
    torch.save(model.state_dict(), save_path)
    log_callback(f"Model saved to {save_path}")
    end_time = time.time()
    elapsed = end_time - start_time
    log_callback(f"Training completed in {elapsed:.2f} seconds")
    return model

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
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("AI Image Training & Prediction Application - v2 Beta - By Shaun Harris")
app.geometry("950x910")

# Create TabView
tabview = ctk.CTkTabview(app, width=900, height=800)
tabview.pack(padx=20, pady=20, fill="both", expand=True)
tabview.add("Train")
tabview.add("Predict")
tabview.add("UI Help")
tabview.add("Model Help")
tabview.set("Train")

try:
    tabview._segmented_button.configure(font=ctk.CTkFont(family="Roboto", size=16))
except Exception as e:
    print("Could not modify segmented button:", e)

# ---------------------------
# Thread-Safe Log Callback
# ---------------------------
def log_callback(message):
    app.after(0, lambda: update_log(message))

def update_log(message):
    log_text.insert("end", f"{message}\n")
    log_text.see("end")

# ---------------------------
# Progress Bar Update Function
# ---------------------------
def update_progress(value):
    percent = int(value * 100)
    app.after(0, lambda: progress_bar.set(value))
    app.after(0, lambda: progress_text_var.set(f"{percent}% Completed"))

# ---------------------------
# TRAIN TAB WIDGETS
# ---------------------------
train_frame = tabview.tab("Train")
modern_font = ctk.CTkFont(family="Roboto", size=14)

good_folder_var = ctk.StringVar()
bad_folder_var = ctk.StringVar()
save_folder_var = ctk.StringVar()

def select_good_folder():
    folder = filedialog.askdirectory(title="Select Good Images Folder")
    if folder:
        good_folder_var.set(folder)

def select_bad_folder():
    folder = filedialog.askdirectory(title="Select Bad Images Folder")
    if folder:
        bad_folder_var.set(folder)

def select_save_folder():
    folder = filedialog.askdirectory(title="Select Folder to Save Model")
    if folder:
        save_folder_var.set(folder)

button_frame = ctk.CTkFrame(train_frame)
button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

roi_toggle_var = ctk.BooleanVar(value=False)
roi_toggle_cb = ctk.CTkCheckBox(button_frame, text="Manual ROI (Region of Interest) Selection - annotate each image with a bounding box", variable=roi_toggle_var, font=modern_font)
roi_toggle_cb.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

good_folder_btn = ctk.CTkButton(button_frame, text="Select Good Images Folder", command=select_good_folder, font=modern_font, width=200)
good_folder_btn.grid(row=1, column=0, padx=5, pady=5, sticky="w")
good_folder_label = ctk.CTkLabel(button_frame, textvariable=good_folder_var, font=modern_font, wraplength=300)
good_folder_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")

bad_folder_btn = ctk.CTkButton(button_frame, text="Select Bad Images Folder", command=select_bad_folder, font=modern_font, width=200)
bad_folder_btn.grid(row=2, column=0, padx=5, pady=5, sticky="w")
bad_folder_label = ctk.CTkLabel(button_frame, textvariable=bad_folder_var, font=modern_font, wraplength=300)
bad_folder_label.grid(row=2, column=1, padx=5, pady=5, sticky="w")

save_folder_btn = ctk.CTkButton(button_frame, text="Select Trained Model Save Location", command=select_save_folder, font=modern_font, width=200)
save_folder_btn.grid(row=3, column=0, padx=5, pady=5, sticky="w")
save_folder_label = ctk.CTkLabel(button_frame, textvariable=save_folder_var, font=modern_font, wraplength=300)
save_folder_label.grid(row=3, column=1, padx=5, pady=5, sticky="w")

params_frame = ctk.CTkFrame(train_frame)
params_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

auto_adjust_var = ctk.BooleanVar(value=True)
auto_adjust_cb = ctk.CTkCheckBox(params_frame, text="Auto Adjust Parameters", variable=auto_adjust_var, font=modern_font)
auto_adjust_cb.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

manual_batch_size_label = ctk.CTkLabel(params_frame, text="Batch Size:", font=modern_font)
manual_epochs_label = ctk.CTkLabel(params_frame, text="Epochs:", font=modern_font)
manual_batch_size_var = ctk.StringVar()
manual_epochs_var = ctk.StringVar()
manual_batch_size_entry = ctk.CTkEntry(params_frame, textvariable=manual_batch_size_var, font=modern_font)
manual_epochs_entry = ctk.CTkEntry(params_frame, textvariable=manual_epochs_var, font=modern_font)

def toggle_manual_entries():
    if auto_adjust_var.get():
        manual_batch_size_label.grid_remove()
        manual_batch_size_entry.grid_remove()
        manual_epochs_label.grid_remove()
        manual_epochs_entry.grid_remove()
    else:
        manual_batch_size_label.grid(row=1, column=0, padx=(0,5), pady=5, sticky="e")
        manual_batch_size_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        manual_epochs_label.grid(row=2, column=0, padx=(0,5), pady=5, sticky="e")
        manual_epochs_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
auto_adjust_cb.configure(command=toggle_manual_entries)
toggle_manual_entries()

log_text = ctk.CTkTextbox(train_frame, width=800, height=200, font=modern_font)
log_text.grid(row=2, column=0, padx=10, pady=10, sticky="w")

progress_text_var = ctk.StringVar(value="0% Completed")
progress_label = ctk.CTkLabel(train_frame, textvariable=progress_text_var, font=modern_font)
progress_label.grid(row=3, column=0, padx=10, pady=(10,0), sticky="w")
progress_bar = ctk.CTkProgressBar(train_frame, width=800)
progress_bar.grid(row=4, column=0, padx=10, pady=10, sticky="w")
progress_bar.set(0)

# Global cancel event for training
cancel_training = threading.Event()

def update_progress(value):
    percent = int(value * 100)
    app.after(0, lambda: progress_bar.set(value))
    app.after(0, lambda: progress_text_var.set(f"{percent}% Completed"))

def start_training():
    cancel_training.clear()  # reset cancel event
    update_progress(0)
    good_folder = good_folder_var.get()
    bad_folder = bad_folder_var.get()
    save_folder = save_folder_var.get()
    # If manual ROI selection is enabled, we require only the folder paths,
    # then we will process each image manually.
    if roi_toggle_var.get():
        log_callback("Manual ROI Selection enabled. Annotating images...")
        data_list = []
        # Process Good images (label 0)
        good_files = load_image_paths(good_folder)
        for file in good_files:
            img = Image.open(file).convert("RGB")
            cropped = select_roi_for_image(img)
            data_list.append((cropped, 0))
        # Process Bad images (label 1)
        bad_files = load_image_paths(bad_folder)
        for file in bad_files:
            img = Image.open(file).convert("RGB")
            cropped = select_roi_for_image(img)
            data_list.append((cropped, 1))
        manual_dataset = ManualROIDataset(data_list, transform=transform)
    else:
        manual_dataset = None

    if not good_folder or (not bad_folder and not roi_toggle_var.get()):
        log_callback("Please select both Good and Bad image folders.")
        return
    log_text.delete("1.0", "end")
    log_callback("Starting training...")
    auto_adjust = auto_adjust_var.get()
    manual_bs = manual_batch_size_var.get() if not auto_adjust else None
    manual_ep = manual_epochs_var.get() if not auto_adjust else None
    train_thread = threading.Thread(
        target=train_model,
        args=(good_folder, bad_folder, save_folder, log_callback, auto_adjust, manual_bs, manual_ep, update_progress, manual_dataset)
    )
    train_thread.start()

# Model Architecture Dropdown & Description
model_arch_var = ctk.StringVar(value="ResNet18")
model_arch_options = ["ResNet18", "ResNet50"]
model_arch_dropdown = ctk.CTkOptionMenu(train_frame, variable=model_arch_var, values=model_arch_options, font=modern_font)
model_arch_dropdown.grid(row=5, column=0, padx=10, pady=10, sticky="nw")

model_arch_descriptions = {
    "ResNet18": "ResNet18: A relatively lightweight neural network with residual connections. Good for general-purpose classification. Use for simple object datasets.",
    "ResNet50": "ResNet50: A deeper model offering higher accuracy on complex tasks but with higher computational cost. Use for more complex datasets. More weighs, 50 in this case, doesn't always mean better performance."
}
def update_model_arch_desc(*args):
    desc = model_arch_descriptions.get(model_arch_var.get(), "")
    model_arch_desc_label.configure(text=desc)
model_arch_var.trace("w", update_model_arch_desc)
model_arch_desc_label = ctk.CTkLabel(train_frame, text=model_arch_descriptions[model_arch_var.get()], font=modern_font, wraplength=300)
model_arch_desc_label.grid(row=6, column=0, padx=10, pady=10, sticky="nw")

train_btn = ctk.CTkButton(train_frame, text="Train Model", command=start_training, font=modern_font, width=150)
train_btn.grid(row=7, column=0, padx=10, pady=10, sticky="w")

cancel_btn = ctk.CTkButton(train_frame, text="Cancel Training", command=lambda: cancel_training.set(), font=modern_font, width=150)
cancel_btn.grid(row=8, column=0, padx=10, pady=10, sticky="w")

##########################
# PREDICT TAB WIDGETS
##########################
predict_frame = tabview.tab("Predict")

# ------------------------------------------------
# 1) Model Selection Container
# ------------------------------------------------
model_select_container = ctk.CTkFrame(predict_frame, corner_radius=10)
model_select_container.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

model_select_label = ctk.CTkLabel(model_select_container, text="Model Selection", font=modern_font)
model_select_label.pack(anchor="nw", padx=5, pady=5)

model_select_frame = ctk.CTkFrame(model_select_container)
model_select_frame.pack(padx=5, pady=5, fill="both", expand=True)

model_path_var = ctk.StringVar()

def select_model_file():
    file_path = filedialog.askopenfilename(title="Select Model (.pth) file", filetypes=[("PyTorch Model", "*.pth")])
    if file_path:
        model_path_var.set(file_path)
        load_model(file_path)

def load_model(file_path):
    global model
    weights = ResNet18_Weights.DEFAULT
    model_loaded = resnet18(weights=weights)
    num_ftrs = model_loaded.fc.in_features
    model_loaded.fc = nn.Linear(num_ftrs, 2)
    model_loaded.load_state_dict(torch.load(file_path, map_location=device))
    model = model_loaded.to(device)
    model.eval()
    predict_status_label.configure(text="Model loaded successfully.", font=modern_font)

select_model_btn = ctk.CTkButton(model_select_frame, text="Select Trained Model File", command=select_model_file, font=modern_font, width=200)
select_model_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

model_file_label = ctk.CTkLabel(model_select_frame, textvariable=model_path_var, font=modern_font, wraplength=300)
model_file_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

# ------------------------------------------------
# 2) Image Prediction Container
# ------------------------------------------------
image_predict_container = ctk.CTkFrame(predict_frame, corner_radius=10)
image_predict_container.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

image_predict_label = ctk.CTkLabel(image_predict_container, text="Image Prediction", font=modern_font)
image_predict_label.pack(anchor="nw", padx=5, pady=5)

predict_image_frame = ctk.CTkFrame(image_predict_container)
predict_image_frame.pack(padx=5, pady=5, fill="both", expand=True)

def select_predict_image():
    file_path = filedialog.askopenfilename(
        title="Select Image For Prediction",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if file_path:
        predicted_class, confidence, img = predict_image(file_path, model)
        if img is None:
            predict_result_label.configure(text="Error loading image.")
            return
        
        display_img = img.resize((250, 250))
        if predicted_class == "bad":
            frame_np = np.array(display_img)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            input_tensor = transform(img).unsqueeze(0).to(device)
            target_layer = model.layer4[1].conv2  # Adjust as needed
            cam = generate_gradcam(model, input_tensor, target_layer)
            cam_resized = cv2.resize(cam, (frame_np.shape[1], frame_np.shape[0]))
            frame_np = overlay_gradcam_boxes(frame_np, cam_resized, threshold=0.5)
            display_img = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
        
        from customtkinter import CTkImage
        ctk_img = CTkImage(light_image=display_img, size=(250, 250))
        predict_image_label.configure(image=ctk_img, text="", font=modern_font)
        predict_image_label.image = ctk_img
        predict_result_label.configure(text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%", font=modern_font)

select_predict_img_btn = ctk.CTkButton(predict_image_frame, text="Select Image For Prediction", command=select_predict_image, font=modern_font, width=200)
select_predict_img_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

predict_image_label = ctk.CTkLabel(predict_image_frame, text="Image Preview", font=modern_font)
predict_image_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

predict_result_label = ctk.CTkLabel(predict_image_frame, text="Prediction will appear here", font=modern_font)
predict_result_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

# ------------------------------------------------
# 3) Video & Screen Capture Prediction Container
# ------------------------------------------------
video_pred_container = ctk.CTkFrame(predict_frame, corner_radius=10)
video_pred_container.grid(row=2, column=0, padx=10, pady=10, sticky="nw")

video_pred_label = ctk.CTkLabel(video_pred_container, text="Live Feed, Recorded Video & Screen Snip", font=modern_font)
video_pred_label.pack(anchor="nw", padx=5, pady=5)

video_pred_frame = ctk.CTkFrame(video_pred_container)
video_pred_frame.pack(padx=5, pady=5, fill="both", expand=True)

def live_feed_prediction():
    if model is None:
        predict_status_label.configure(text="Error: No model loaded.", font=modern_font)
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        predict_status_label.configure(text="Error: Unable to access camera", font=modern_font)
        return
    target_layer = model.layer4[1].conv2
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            max_prob, pred = torch.max(probs, 1)
            confidence = max_prob.item() * 100
            predicted_class = "good" if pred.item() == 0 else "bad"
            color = (0,255,0) if predicted_class=="good" else (0,0,255)
            cv2.putText(frame, f"{predicted_class}: {confidence:.2f}%", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if predicted_class == "bad":
            cam = generate_gradcam(model, input_tensor, target_layer)
            cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
            frame = overlay_gradcam_boxes(frame, cam_resized, threshold=0.5)
        cv2.imshow("Live Feed Prediction (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def recorded_video_prediction():
    if model is None:
        predict_status_label.configure(text="Error: No model loaded.", font=modern_font)
        return
    video_path = filedialog.askopenfilename(title="Select Recorded Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        predict_status_label.configure(text="Error: Unable to open video file", font=modern_font)
        return
    target_layer = model.layer4[1].conv2
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            max_prob, pred = torch.max(probs, 1)
            confidence = max_prob.item() * 100
            predicted_class = "good" if pred.item() == 0 else "bad"
            color = (0,255,0) if predicted_class=="good" else (0,0,255)
            cv2.putText(frame, f"{predicted_class}: {confidence:.2f}%", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if predicted_class == "bad":
            cam = generate_gradcam(model, input_tensor, target_layer)
            cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
            frame = overlay_gradcam_boxes(frame, cam_resized, threshold=0.5)
        cv2.imshow("Recorded Video Prediction (Press 'q' to quit)", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def screen_capture_prediction():
    if model is None:
        predict_status_label.configure(text="Error: No model loaded.", font=modern_font)
        return
    app.after(0, snip_screen_prediction)

# ---------------------------
# Screen Snip Prediction Function (with Grad-CAM)
# ---------------------------
def snip_screen_prediction():
    # Create a full-screen transparent Toplevel for snipping
    top = tk.Toplevel(app)
    top.attributes("-fullscreen", True)
    top.attributes("-alpha", 0.3)
    top.configure(background="black")
    
    canvas = tk.Canvas(top, cursor="cross", bg="gray")
    canvas.pack(fill=tk.BOTH, expand=True)
    
    start_x, start_y = None, None
    rect = None

    def on_button_press(event):
        nonlocal start_x, start_y, rect
        start_x, start_y = event.x, event.y
        rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="red", width=2)

    def on_move_press(event):
        nonlocal rect
        curX, curY = (event.x, event.y)
        canvas.coords(rect, start_x, start_y, curX, curY)

    def on_button_release(event):
        nonlocal start_x, start_y, rect
        end_x, end_y = event.x, event.y
        top.destroy()
        bbox = (min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y))
        screen = ImageGrab.grab(bbox=bbox)
        screen = screen.convert("RGB")
        input_tensor = transform(screen).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            max_prob, pred = torch.max(probs, 1)
            confidence = max_prob.item() * 100
            class_names = ["good", "bad"]
            predicted_class = class_names[pred.item()]
        screen_np = np.array(screen)
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
        if predicted_class == "bad":
            target_layer = model.layer4[1].conv2
            cam = generate_gradcam(model, input_tensor, target_layer)
            cam_resized = cv2.resize(cam, (screen_np.shape[1], screen_np.shape[0]))
            screen_np = overlay_gradcam_boxes(screen_np, cam_resized, threshold=0.5)
        color = (0,255,0) if predicted_class == "good" else (0,0,255)
        cv2.putText(screen_np, f"{predicted_class}: {confidence:.2f}%", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Screen Snip Prediction (Press 'q' to quit)", screen_np)
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)
    top.mainloop()

live_feed_btn = ctk.CTkButton(video_pred_frame, text="Live Feed Prediction", command=lambda: threading.Thread(target=live_feed_prediction).start(), font=modern_font, width=200)
live_feed_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

recorded_video_btn = ctk.CTkButton(video_pred_frame, text="Recorded Video Prediction", command=lambda: threading.Thread(target=recorded_video_prediction).start(), font=modern_font, width=200)
recorded_video_btn.grid(row=0, column=1, padx=5, pady=5, sticky="w")

screen_capture_btn = ctk.CTkButton(video_pred_frame, text="Screen Snip Prediction", command=lambda: threading.Thread(target=screen_capture_prediction).start(), font=modern_font, width=200)
screen_capture_btn.grid(row=1, column=0, padx=5, pady=5, sticky="w")

predict_status_label = ctk.CTkLabel(predict_frame, text="", font=modern_font)
predict_status_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

##########################
# UI HELP TAB WIDGETS
##########################
help_frame_ui = tabview.tab("UI Help")

help_frame_inner = ctk.CTkFrame(help_frame_ui)
help_frame_inner.pack(padx=10, pady=10, fill="both", expand=True)

help_font = ctk.CTkFont(family="Roboto", size=14)

help_textbox = ctk.CTkTextbox(help_frame_inner, font=help_font, wrap="word")
help_textbox.pack(padx=10, pady=10, fill="both", expand=True)
help_text = (
    "TRAIN TAB:\n"
    "- Select the folder containing 'Good' images and the folder containing 'Bad' images (for two-class mode).\n"
    "- Allowed image formats: jpg, jpeg, png, bmp.\n"
    "- Optionally, select a folder where the trained model will be saved.\n"
    "- Use the drop-down menu to select the model architecture. For example, ResNet18 is lightweight and good for general tasks, whereas ResNet50 is deeper and may provide higher accuracy on complex datasets.\n"
    "- By default, the application auto-adjusts training parameters. If you disable auto-adjust, you can manually enter the Batch Size and Epochs.\n\n"
    "PREDICT TAB:\n"
    "- Load a model (.pth file) and then select an image for prediction.\n"
    "- You can also use Live Feed, Recorded Video, or Screen Snip prediction options.\n"
    "- If the model predicts 'bad', the regions contributing to that decision will be highlighted with bounding boxes.\n\n"
    "TRAINING TERMINOLOGY:\n"
    "- Epoch: One complete pass through the training data.\n"
    "- Loss: A measure of error; lower values indicate better performance.\n"
    "- Validation Accuracy: The percentage of correctly classified images on unseen data.\n\n"
    "This application uses transfer learning with a pre-trained ResNet model. Training runs in a separate thread so that the UI remains responsive, and a progress bar shows training completion."
)
help_textbox.insert("0.0", help_text)
help_textbox.configure(state="disabled")

##########################
# MODEL HELP TAB WIDGETS
##########################
help_frame_model = tabview.tab("Model Help")

detailed_container = ctk.CTkFrame(help_frame_model)
detailed_container.pack(padx=5, pady=5, fill="both", expand=True)

modelhelp_font = ctk.CTkFont(family="Roboto", size=12)

detailed_label = ctk.CTkLabel(detailed_container, text="Here’s how the model works and what happens during training", font=modelhelp_font)
detailed_label.pack(anchor="w", padx=5, pady=(5, 0))

detailed_frame = ctk.CTkFrame(detailed_container, height=400)
detailed_frame.pack(padx=5, pady=5, fill="both", expand=True)
detailed_frame.pack_propagate(False)

detailed_textbox = ctk.CTkTextbox(detailed_frame, font=modelhelp_font, wrap="word")
detailed_textbox.pack(padx=5, pady=5, fill="both", expand=True)

detailed_text = (
    "Decision Process (Inference)\n\n"
    "Feature Extraction:\n"
    "The model, built on a convolutional neural network (CNN) like ResNet, processes an image through several layers that automatically learn to recognize useful features—such as edges, textures, and more complex patterns. These features help the model distinguish between “good” and “bad” images.\n\n"
    "Classification Head:\n"
    "After feature extraction, the final layers (usually fully connected layers) take these learned features and assign scores to each class. These scores are then converted into probabilities (using a function like softmax). The class with the highest probability becomes the model's prediction.\n\n"
    "What Happens During Training\n"
    "Forward Pass:\n"
    "The training images (labeled as “good” or “bad”) are fed into the model. The model makes predictions based on its current weights (its internal parameters).\n\n"
    "Loss Calculation:\n"
    "A loss function (such as cross-entropy loss) measures the difference between the model’s predictions and the actual labels. For example, if the model predicts 70% 'good' for an image that is actually 'good', the loss will be relatively low. If it predicts incorrectly, the loss will be higher.\n\n"
    "Backpropagation:\n"
    "The model then calculates how much each weight contributed to the loss. It uses this information to adjust (or 'update') the weights, aiming to reduce the loss in future predictions.\n\n"
    "Epochs and Iteration:\n"
    "One epoch is a complete pass through the entire training dataset. The training process repeats for many epochs (e.g., 42 epochs), during which the model gradually improves its performance. You’ll typically see the loss decrease and the accuracy increase over time.\n\n"
)

detailed_textbox.insert("0.0", detailed_text)
detailed_textbox.configure(state="disabled")

simple_container = ctk.CTkFrame(help_frame_model)
simple_container.pack(padx=5, pady=5, fill="both", expand=True)

simple_label = ctk.CTkLabel(simple_container, text="In Simple Terms", font=modelhelp_font)
simple_label.pack(anchor="w", padx=5, pady=(5, 0))

simple_frame = ctk.CTkFrame(simple_container, height=300)
simple_frame.pack(padx=5, pady=5, fill="both", expand=True)
simple_frame.pack_propagate(False)

simple_textbox = ctk.CTkTextbox(simple_frame, font=modelhelp_font, wrap="word")
simple_textbox.pack(padx=5, pady=5, fill="both", expand=True)

simple_text = (
    "Before Training:\n"
    "The model starts with random weights—it doesn’t “know” what good or bad images look like.\n\n"
    "During Training:\n"
    "It looks at many examples of good and bad images, compares its guesses to the actual labels, and learns by tweaking its internal settings to get better at distinguishing between the two.\n\n"
    "After Training:\n"
    "The model has “learned” the patterns that define a good image versus a bad one. When you feed it a new image, it extracts features, processes them through the layers it trained, and outputs a decision based on the patterns it has learned."
)

simple_textbox.insert("0.0", simple_text)
simple_textbox.configure(state="disabled")

# ---------------------------
# Run the Application
# ---------------------------
log_callback("Welcome to the AI Image Training & Prediction Application!")
log_callback("Select Good & Bad image Folders")
log_callback("Select Trained Model Saved Output Location")
log_callback("Awaiting User Input...")
app.mainloop()
