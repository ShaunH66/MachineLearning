import os
import time
import threading
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

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
# Training Function with Progress Callback
# ---------------------------
def train_model(good_folder, bad_folder, save_folder, log_callback,
                auto_adjust=True, manual_batch_size=None, manual_epochs=None,
                progress_callback=None):
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
        # Update progress bar (value from 0.0 to 1.0)
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
    
    # Save model to the selected folder.
    if not save_folder:
        save_folder = os.getcwd()  # default to current directory
    save_path = os.path.join(save_folder, "user_trained_model.pth")
    torch.save(model.state_dict(), save_path)
    log_callback(f"Model saved to {save_path}")
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
app.title("AI Image Training & Prediction Application - By Shaun Harris")
app.geometry("950x750")

# Create TabView
tabview = ctk.CTkTabview(app, width=900, height=700)
tabview.pack(padx=20, pady=20, fill="both", expand=True)
tabview.add("Train")
tabview.add("Predict")
tabview.add("Help")
tabview.set("Train")  # default active tab

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
    app.after(0, lambda: progress_bar.set(value))

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

# Vertical layout frame for folder selection
button_frame = ctk.CTkFrame(train_frame)
button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

good_folder_btn = ctk.CTkButton(button_frame, text="Select Good Images Folder", command=select_good_folder, font=modern_font, width=200)
good_folder_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")
good_folder_label = ctk.CTkLabel(button_frame, textvariable=good_folder_var, font=modern_font, wraplength=300)
good_folder_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

bad_folder_btn = ctk.CTkButton(button_frame, text="Select Bad Images Folder", command=select_bad_folder, font=modern_font, width=200)
bad_folder_btn.grid(row=1, column=0, padx=5, pady=5, sticky="w")
bad_folder_label = ctk.CTkLabel(button_frame, textvariable=bad_folder_var, font=modern_font, wraplength=300)
bad_folder_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")

save_folder_btn = ctk.CTkButton(button_frame, text="Select Trained Model Save Location", command=select_save_folder, font=modern_font, width=200)
save_folder_btn.grid(row=2, column=0, padx=5, pady=5, sticky="w")
save_folder_label = ctk.CTkLabel(button_frame, textvariable=save_folder_var, font=modern_font, wraplength=300)
save_folder_label.grid(row=2, column=1, padx=5, pady=5, sticky="w")

# Parameter frame with auto-adjust checkbox and manual inputs
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

# Log textbox for training progress
log_text = ctk.CTkTextbox(train_frame, width=800, height=200, font=modern_font)
log_text.grid(row=2, column=0, padx=10, pady=10, sticky="w")

# Create a StringVar to hold the progress percentage text
progress_text_var = ctk.StringVar(value="0% Completed")

# Label above the progress bar showing progress percentage
progress_label = ctk.CTkLabel(train_frame, textvariable=progress_text_var, font=modern_font)
progress_label.grid(row=3, column=0, padx=10, pady=(10,0), sticky="w")

# Progress Bar (0.0 to 1.0)
progress_bar = ctk.CTkProgressBar(train_frame, width=800)
progress_bar.grid(row=4, column=0, padx=10, pady=10, sticky="w")
progress_bar.set(0)

# Progress update function to also update the label text
def update_progress(value):
    # value is between 0 and 1; convert to percentage
    percent = int(value * 100)
    app.after(0, lambda: progress_bar.set(value))
    app.after(0, lambda: progress_text_var.set(f"{percent}% Completed"))

def start_training():
    good_folder = good_folder_var.get()
    bad_folder = bad_folder_var.get()
    save_folder = save_folder_var.get()
    if not good_folder or not bad_folder:
        log_callback("Please select both Good and Bad image folders.")
        return
    log_text.delete("1.0", "end")
    log_callback("Starting training...")
    auto_adjust = auto_adjust_var.get()
    manual_bs = manual_batch_size_var.get() if not auto_adjust else None
    manual_ep = manual_epochs_var.get() if not auto_adjust else None
    # Start training in a new thread so the GUI remains responsive.
    train_thread = threading.Thread(
        target=train_model,
        args=(good_folder, bad_folder, save_folder, log_callback, auto_adjust, manual_bs, manual_ep, update_progress)
    )
    train_thread.start()

train_btn = ctk.CTkButton(train_frame, text="Train Model", command=start_training, font=modern_font, width=150)
train_btn.grid(row=5, column=0, padx=10, pady=10, sticky="w")

# ---------------------------
# Predict TAB WIDGETS
# ---------------------------
predict_frame = tabview.tab("Predict")

# Frame for model selection
model_select_frame = ctk.CTkFrame(predict_frame)
model_select_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

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

# Frame for predict image selection and preview
predict_image_frame = ctk.CTkFrame(predict_frame)
predict_image_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

def select_predict_image():
    file_path = filedialog.askopenfilename(title="Select Image For Prediction", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        predicted_class, confidence, img = predict_image(file_path, model)
        if img is None:
            predict_result_label.configure(text="Error loading predict image.")
            return
        display_img = img.resize((250, 250))
        from customtkinter import CTkImage
        ctk_img = CTkImage(light_image=display_img, size=(250, 250))
        predict_image_label.configure(image=ctk_img, text="", font=modern_font)
        predict_image_label.image = ctk_img
        predict_result_label.configure(text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%", font=modern_font)

select_predict_img_btn = ctk.CTkButton(predict_image_frame, text="Select Image For Prediction", command=select_predict_image, font=modern_font, width=200)
select_predict_img_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

predict_image_label = ctk.CTkLabel(predict_image_frame, text="Image Preview", font=modern_font)
predict_image_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

predict_result_frame = ctk.CTkFrame(predict_frame)
predict_result_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nw")

predict_result_label = ctk.CTkLabel(predict_result_frame, text="Prediction will appear here", font=modern_font)
predict_result_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

predict_status_label = ctk.CTkLabel(predict_result_frame, text="", font=modern_font)
predict_status_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

# ---------------------------
# HELP TAB WIDGETS
# ---------------------------
help_frame = tabview.tab("Help")

help_frame_inner = ctk.CTkFrame(help_frame)
help_frame_inner.pack(padx=10, pady=10, fill="both", expand=True)
ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
help_text = (
    "TRAIN TAB:\n"
    "- Select the folder containing 'Good' images and the folder containing 'Bad' images.\n"
    "- Allowed image formats - jpg, jpeg, png, bmp.\n"
    "- Optionally, select a folder where the trained model will be saved.\n"
    "- By default, the application auto-adjusts training parameters:\n"
    "    • Batch Size = min(32, max(4, train_size//10))\n"
    "    • Epochs = max(10, min(50, total_images//10))\n\n"
    "If you disable auto-adjust, you can manually enter the Batch Size and Epochs.\n\n"
    "Predict TAB:\n"
    "- Load a model (.pth file) and then select a test image.\n"
    "- The model's prediction and confidence will be displayed.\n\n"
    "This application uses transfer learning with a pre-trained ResNet18.\n"
    "Training runs in a separate thread so that the GUI remains responsive.\n"
    "A progress bar displays training progression."
)

help_label = ctk.CTkLabel(help_frame_inner, text=help_text, justify="left", wraplength=850, font=modern_font)
help_label.pack(padx=10, pady=10, anchor="nw")

# ---------------------------
# Run the Application
# ---------------------------
app.mainloop()
