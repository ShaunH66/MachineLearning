import os
import customtkinter as ctk
from customtkinter import CTkFont
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
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
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
# Training Function
# ---------------------------
def train_model(good_folder, bad_folder, save_folder, log_callback,
                auto_adjust=True, manual_batch_size=None, manual_epochs=None):
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
        for inputs, labels in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
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
    
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in DataLoader(val_dataset, batch_size=batch_size, shuffle=False):
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
# GUI Using CustomTkinter
# ---------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("AI Training & Prediction Application - By Shaun Harris")
app.geometry("800x650")

# Create the tabview and add tabs first
tabview = ctk.CTkTabview(app, width=600, height=700)
tabview.pack(padx=20, pady=20, fill="both", expand=True)

tabview.add("Train")
tabview.add("Predict")
tabview.add("Help")
tabview.set("Train")  # Set default active tab

tabview.configure(corner_radius=10)

try:
    tabview._segmented_button.configure(font=CTkFont(family="Roboto", size=16))
except Exception as e:
    print("Could not modify the underlying segmented button:", e)

# ---------------------------
# TRAIN TAB WIDGETS
# ---------------------------
train_frame = tabview.tab("Train")

# Define a more modern font
modern_font = ctk.CTkFont(family="Roboto", size=14)

# Variables to store selected folders
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

# A frame to group the top three folder-selection buttons vertically
button_frame = ctk.CTkFrame(train_frame)
button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

# "Select Good Images Folder" button + label
good_folder_btn = ctk.CTkButton(
    button_frame, text="Select Good Images Folder",
    command=select_good_folder, font=modern_font, width=200
)
good_folder_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

good_folder_label = ctk.CTkLabel(
    button_frame, textvariable=good_folder_var, font=modern_font, wraplength=300
)
good_folder_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

# "Select Bad Images Folder" button + label
bad_folder_btn = ctk.CTkButton(
    button_frame, text="Select Bad Images Folder",
    command=select_bad_folder, font=modern_font, width=200
)
bad_folder_btn.grid(row=1, column=0, padx=5, pady=5, sticky="w")

bad_folder_label = ctk.CTkLabel(
    button_frame, textvariable=bad_folder_var, font=modern_font, wraplength=300
)
bad_folder_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")

# "Select Save Folder" button + label
save_folder_btn = ctk.CTkButton(
    button_frame, text="Select Model Save Location",
    command=select_save_folder, font=modern_font, width=200
)
save_folder_btn.grid(row=2, column=0, padx=5, pady=5, sticky="w")

save_folder_label = ctk.CTkLabel(
    button_frame, textvariable=save_folder_var, font=modern_font, wraplength=300
)
save_folder_label.grid(row=2, column=1, padx=5, pady=5, sticky="w")

# A frame for the auto-adjust checkbox and parameter inputs
params_frame = ctk.CTkFrame(train_frame)
params_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

auto_adjust_var = ctk.BooleanVar(value=True)
auto_adjust_cb = ctk.CTkCheckBox(
    params_frame, text="Auto Adjust Train Parameters Based Off Total Number Of Images Provided",
    variable=auto_adjust_var, font=modern_font
)
auto_adjust_cb.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

# Labels and entries for manual parameters (hidden by default)
manual_batch_size_label = ctk.CTkLabel(params_frame, text="Batch Size:", font=modern_font)
manual_epochs_label = ctk.CTkLabel(params_frame, text="Epochs:", font=modern_font)

manual_batch_size_var = ctk.StringVar()
manual_epochs_var = ctk.StringVar()

manual_batch_size_entry = ctk.CTkEntry(params_frame, textvariable=manual_batch_size_var, font=modern_font)
manual_epochs_entry = ctk.CTkEntry(params_frame, textvariable=manual_epochs_var, font=modern_font)

def toggle_manual_entries():
    """ Show/hide manual parameter widgets based on checkbox state. """
    if auto_adjust_var.get():
        # Hide manual parameter widgets
        manual_batch_size_label.grid_remove()
        manual_epochs_label.grid_remove()
        manual_batch_size_entry.grid_remove()
        manual_epochs_entry.grid_remove()
    else:
        # Show manual parameter widgets
        manual_batch_size_label.grid(row=1, column=0, padx=(0,5), pady=5, sticky="e")
        manual_batch_size_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        manual_epochs_label.grid(row=2, column=0, padx=(0,5), pady=5, sticky="e")
        manual_epochs_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

auto_adjust_cb.configure(command=toggle_manual_entries)
toggle_manual_entries()

# A textbox to log training progress
log_text = ctk.CTkTextbox(train_frame, width=720, height=200, font=modern_font)
log_text.grid(row=2, column=0, padx=10, pady=10, sticky="w")

def log_callback(message):
    log_text.insert("end", f"{message}\n")
    log_text.see("end")
    app.update_idletasks()

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

    train_model(
        good_folder, bad_folder, save_folder,
        log_callback, auto_adjust,
        manual_bs, manual_ep
    )

train_btn = ctk.CTkButton(
    train_frame, text="Train Model",
    command=start_training, font=modern_font, width=150
)
train_btn.grid(row=3, column=0, padx=10, pady=10, sticky="w")

# ---------------------------
# PREDICT TAB WIDGETS
# ---------------------------
test_frame = tabview.tab("Predict")

# A frame to hold the model-selection widgets in a neat layout
model_select_frame = ctk.CTkFrame(test_frame)
model_select_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

model_path_var = ctk.StringVar()

def select_model_file():
    file_path = filedialog.askopenfilename(
        title="Select Model (.pth) file",
        filetypes=[("PyTorch Model", "*.pth")]
    )
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
    test_status_label.configure(text="Model loaded successfully.", font=modern_font)

select_model_btn = ctk.CTkButton(
    model_select_frame,
    text="Select Model File",
    command=select_model_file,
    font=modern_font,
    width=200
)
select_model_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

model_file_label = ctk.CTkLabel(
    model_select_frame,
    textvariable=model_path_var,
    font=modern_font,
    wraplength=300
)
model_file_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

# A frame for test image selection and preview
test_image_frame = ctk.CTkFrame(test_frame)
test_image_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

def select_test_image():
    file_path = filedialog.askopenfilename(
        title="Select Test Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if file_path:
        predicted_class, confidence, img = predict_image(file_path, model)
        if img is None:
            test_result_label.configure(text="Error loading test image.")
            return
        display_img = img.resize((250, 250))
        from customtkinter import CTkImage
        ctk_img = CTkImage(light_image=display_img, size=(250, 250))
        test_image_label.configure(image=ctk_img, text="", font=modern_font)
        test_image_label.image = ctk_img
        test_result_label.configure(
            text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%",
            font=modern_font
        )

select_test_img_btn = ctk.CTkButton(
    test_image_frame,
    text="Select Test Image",
    command=select_test_image,
    font=modern_font,
    width=200
)
select_test_img_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

# Image preview label
test_image_label = ctk.CTkLabel(test_image_frame, text="Image Preview", font=modern_font)
test_image_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

# A frame for result/status
test_result_frame = ctk.CTkFrame(test_frame)
test_result_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nw")

test_result_label = ctk.CTkLabel(
    test_result_frame,
    text="Prediction will appear here",
    font=modern_font
)
test_result_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

test_status_label = ctk.CTkLabel(
    test_result_frame,
    text="",
    font=modern_font
)
test_status_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

# ---------------------------
# HELP TAB WIDGETS
# ---------------------------
help_frame = tabview.tab("Help")

help_frame_inner = ctk.CTkFrame(help_frame)
help_frame_inner.pack(padx=10, pady=10, fill="both", expand=True)
('.jpg', '.jpeg', '.png', '.bmp')
help_text = (
    "TRAIN TAB:\n\n"
    "- Select the folder containing 'Good' images and the folder containing 'Bad' images.\n\n"
    "- Allowed image formats - jpg, jpeg, png, bmp.\n\n"
    "- Optionally, select a folder where the trained model will be saved.\n\n"
    "- By default, the application auto-adjusts training parameters:\n"
    "    • Batch Size = min(32, max(4, train_size//10))\n"
    "    • Epochs = max(10, min(50, total_images//10))\n\n"
    "PREDICT TAB:\n\n"
    "- Load a model (.pth file) and then select a test image.\n\n"
    "- The model's prediction and confidence will be displayed.\n\n"
)

help_label = ctk.CTkLabel(
    help_frame_inner,
    text=help_text,
    justify="left",
    wraplength=850,
    font=modern_font
)
help_label.pack(padx=10, pady=10, anchor="nw")

# ---------------------------
# Run the Application
# ---------------------------
app.mainloop()
