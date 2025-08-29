import os
import random
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import time
from datetime import datetime
import warnings
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from shutil import copyfile
import torch.nn.functional as F



# === CONFIGURATION VARIABLES ===
DATA_DIR = "/home/ubuntu/Images/"
ZIP_PATH = "/home/ubuntu/ToN1.zip"
TRAIN_SIZE=0.7
TEST_SIZE=0.15
VAL_SIZE=0.15
EPOCH=100
#===============================#
# Ignore warnings
warnings.filterwarnings("ignore")

# GPU Memory Monitoring Helper
class GPUMemoryMonitor:
    def __init__(self):
        self.memory_log = []
        self.gpu_available = torch.cuda.is_available()

    def _get_gpu_memory(self):
        if not self.gpu_available:
            return None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = info.total / (1024 ** 2)
            used = info.used / (1024 ** 2)
            return {
                'total_mb': total,
                'used_mb': used,
                'free_mb': total - used,
                'utilization_%': (used / total) * 100
            }
        except Exception:
            return {
                'message': 'Use pynvml for detailed GPU info'
            }

    def log(self, stage, epoch=None, additional=None):
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024 ** 3)
        ram_total = ram.total / (1024 ** 3)
        gpu_info = self._get_gpu_memory()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'stage': stage,
            'epoch': epoch,
            'ram_used': ram_used,
            'ram_total': ram_total,
            'ram_percent': ram.percent,
            'gpu_info': gpu_info,
            'additional': additional
        }
        self.memory_log.append(log_entry)
        print(f"[{timestamp}] {stage}" + (f" (Epoch {epoch})" if epoch else ""))
        print(f"RAM: {ram_used:.2f}GB/{ram_total:.2f}GB ({ram.percent}%)")
        if gpu_info:
            if 'used_mb' in gpu_info:
                print(f"GPU: {gpu_info['used_mb']:.0f}MB/{gpu_info['total_mb']:.0f}MB ({gpu_info['utilization_%']:.1f}%)")
            else:
                print(f"GPU: {gpu_info.get('message', 'Unknown')}")
        if additional:
            print(f"Info: {additional}")
        print("-" * 50)

    def plot_usage(self):
        if not self.memory_log:
            print("No memory log available.")
            return
        ram_usage = [log['ram_percent'] for log in self.memory_log]
        timestamps = [log['timestamp'] for log in self.memory_log]
        stages = [log['stage'] for log in self.memory_log]
        gpu_usage = [log['gpu_info']['used_mb'] / log['gpu_info']['total_mb'] * 100 if log['gpu_info'] and 'used_mb' in log['gpu_info'] else 0 for log in self.memory_log]
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        plt.plot(ram_usage, 'b-o', label='RAM Usage')
        plt.title('RAM Usage Over Time')
        plt.ylabel('RAM Usage (%)')
        plt.grid(True)
        for i, stage in enumerate(stages):
            plt.annotate(stage, (i, ram_usage[i]), rotation=45, fontsize=8, ha='center', textcoords="offset points", xytext=(0,10))
        plt.subplot(2, 1, 2)
        plt.plot(gpu_usage, 'r-o', label='GPU Usage')
        plt.title('GPU Usage Over Time')
        plt.ylabel('GPU Usage (%)')
        plt.xlabel('Time Steps')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def summary(self):
        ram_usage = [log['ram_percent'] for log in self.memory_log]
        print("="*60)
        print("MEMORY USAGE SUMMARY")
        print("="*60)
        print(f"RAM - Max: {max(ram_usage):.1f}%, Min: {min(ram_usage):.1f}%, Avg: {np.mean(ram_usage):.1f}%")
        gpu_usage = [log['gpu_info']['used_mb'] / log['gpu_info']['total_mb'] * 100 for log in self.memory_log if log['gpu_info'] and 'used_mb' in log['gpu_info']]
        if gpu_usage:
            print(f"GPU - Max: {max(gpu_usage):.1f}%, Min: {min(gpu_usage):.1f}%, Avg: {np.mean(gpu_usage):.1f}%")
        print("="*60)

# CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(CNN, self).__init__()
        self.num_classes = num_classes

        # CNN Block 1: Input 150x150
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: 75x75
        self.dropout1 = nn.Dropout2d(0.25)

        # CNN Block 2: Input 75x75
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: 37x37
        self.dropout2 = nn.Dropout2d(0.25)

        # CNN Block 3: Input 37x37
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: 18x18
        self.dropout3 = nn.Dropout2d(0.25)

        # CNN Block 4: Input 18x18
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # Output: 9x9
        self.dropout4 = nn.Dropout2d(0.25)

        # Calculate flattened size
        self.fc_input_size = 256 * 9 * 9  # 20,736 features

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)

        return x

# Main Dynamic CNN Class
class DynamicCNN:
    def __init__(self, num_classes=None, data_dir=DATA_DIR, train_split=TRAIN_SIZE, val_split=VAL_SIZE, test_split=TEST_SIZE):
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.model = None
        self.class_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_monitor = GPUMemoryMonitor()
        self.memory_monitor.log("Initialization")

    def check_system_resources(self):
        print("\n" + "="*50)
        print("SYSTEM RESOURCE CHECK")
        print("="*50)
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("No GPU available. Using CPU.")
        ram = psutil.virtual_memory()
        print(f"RAM: {ram.total / 1e9:.2f} GB")
        self.memory_monitor.log("Resource Check")

    def extract_data(self, zip_path):
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print(f"Data extracted from {zip_path}")
        else:
            print(f"Zip file not found: {zip_path}")

    def discover_classes(self, source_dir):
        if not os.path.exists(source_dir):
            print(f"Source directory not found: {source_dir}")
            return
        all_items = os.listdir(source_dir)
        data_dirs = [d for d in all_items if os.path.isdir(os.path.join(source_dir, d)) and d.startswith('data_')]
        self.class_names = [d.replace('data_', '') for d in data_dirs]
        self.class_names.sort()
        if self.num_classes is None:
            self.num_classes = len(self.class_names)
        print(f"Detected {self.num_classes} classes: {self.class_names}")

    def split_data(self, source_dir, train_dir, val_dir, test_dir):
        files = [f for f in os.listdir(source_dir) if os.path.getsize(os.path.join(source_dir, f)) > 0]
        random.shuffle(files)
        train_end = int(len(files) * self.train_split)
        val_end = train_end + int(len(files) * self.val_split)
        for f in files[:train_end]: copyfile(os.path.join(source_dir, f), os.path.join(train_dir, f))
        for f in files[train_end:val_end]: copyfile(os.path.join(source_dir, f), os.path.join(val_dir, f))
        for f in files[val_end:]: copyfile(os.path.join(source_dir, f), os.path.join(test_dir, f))
        print(f"Split {len(files)} files: {train_end} train, {val_end - train_end} val, {len(files) - val_end} test")

    def prepare_data(self, source_base_dir):
        self.discover_classes(source_base_dir)
        for cls in self.class_names:
            os.makedirs(f"{self.data_dir}/training/{cls}", exist_ok=True)
            os.makedirs(f"{self.data_dir}/validation/{cls}", exist_ok=True)
            os.makedirs(f"{self.data_dir}/testing/{cls}", exist_ok=True)
            source = f"{source_base_dir}/data_{cls}"
            self.split_data(source, f"{self.data_dir}/training/{cls}", f"{self.data_dir}/validation/{cls}", f"{self.data_dir}/testing/{cls}")

    def build_model(self):
        self.model = CNN(self.num_classes).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss() if self.num_classes == 2 else nn.CrossEntropyLoss()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-4)
        print(f"Model built for {self.num_classes} classes")

    def create_loaders(self, batch_size=32, target_size=(150, 150)):
        transform_train = transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_val = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_data = datasets.ImageFolder(f"{self.data_dir}/training/", transform=transform_train)
        val_data = datasets.ImageFolder(f"{self.data_dir}/validation/", transform=transform_val)
        test_data = datasets.ImageFolder(f"{self.data_dir}/testing/", transform=transform_val)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        print(f"Found {len(train_data)} training images")

    def train(self, epochs=EPOCH):
        self.model.train()
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        for epoch in range(epochs):
            self.memory_monitor.log("Training", epoch+1)
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate_one_epoch()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            self.memory_monitor.log("Epoch End", epoch+1, additional=f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Val_Acc: {val_acc:.2f}%")
        return history

    def _train_one_epoch(self):
        total_loss = 0
        correct = 0
        total = 0
        self.model.train()
        for images, labels in tqdm(self.train_loader, desc='Training'):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            preds = torch.sigmoid(outputs).round() if self.num_classes == 2 else outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        return total_loss / len(self.train_loader), 100 * correct / total

    def _validate_one_epoch(self):
        total_loss = 0
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.sigmoid(outputs).round() if self.num_classes == 2 else outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        return total_loss / len(self.val_loader), 100 * correct / total

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = torch.sigmoid(outputs).round() if self.num_classes == 2 else outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.show()
        print(classification_report(all_labels, all_preds, target_names=self.class_names))

    def plot_history(self, history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.legend()
        plt.title('Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')
        plt.show()

# Main Function
def main():
    print("Starting PyTorch Dynamic CNN with GPU Monitoring")
    classifier = DynamicCNN(num_classes=None, train_split=TRAIN_SIZE, val_split=VAL_SIZE, test_split=TEST_SIZE)
    classifier.check_system_resources()
    classifier.extract_data(ZIP_PATH)
    classifier.prepare_data(DATA_DIR)
    classifier.build_model()
    classifier.create_loaders()
    history = classifier.train(epochs=EPOCH)
    classifier.evaluate()
    classifier.plot_history(history)
    classifier.memory_monitor.plot_usage()
    classifier.memory_monitor.summary()

if __name__ == "__main__":
    main()