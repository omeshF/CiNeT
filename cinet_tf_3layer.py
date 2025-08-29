import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import psutil
import time
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

# === CONFIGURATION VARIABLES ===
DATA_DIR = "/home/ubuntu/Images/"
#DATA_DIR = "/content/Images/"
#ZIP_PATH = "/content/ToN1.zip"
ZIP_PATH = "/home/ubuntu/ToN1.zip"
TRAIN_SIZE=0.7
TEST_SIZE=0.15
VAL_SIZE=0.15
EPOCH=100
#===============================#


class GPUMemoryMonitor:
    """Helper class to monitor GPU memory usage"""
    
    def __init__(self):
        self.memory_log = []
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self):
        """Check if GPU is available and accessible"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                print(f"Found {len(gpus)} GPU(s)")
                for i, gpu in enumerate(gpus):
                    print(f"GPU {i}: {gpu}")
                return True
            else:
                print("No GPU found")
                return False
        except Exception as e:
            print(f"Error checking GPU: {e}")
            return False
    
    def get_gpu_memory_info(self):
        """Get current GPU memory usage"""
        if not self.gpu_available:
            return None
            
        try:
            # Get GPU memory info using nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                total_mb = mem_info.total / 1024**2
                used_mb = mem_info.used / 1024**2
                free_mb = mem_info.free / 1024**2
                
                return {
                    'total_mb': total_mb,
                    'used_mb': used_mb,
                    'free_mb': free_mb,
                    'utilization_%': (used_mb / total_mb) * 100
                }
            except ImportError:
                # Fallback to tensorflow memory info
                try:
                    # Get TensorFlow GPU memory info
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    if gpus:
                        # This is a simplified approach - actual memory usage may vary
                        return {
                            'message': 'GPU detected but detailed memory info requires pynvml package',
                            'gpu_count': len(gpus)
                        }
                except:
                    pass
                    
        except Exception as e:
            print(f"Error getting GPU memory info: {e}")
            
        return None
    
    def log_memory_usage(self, stage, epoch=None, additional_info=None):
        """Log current memory usage"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get GPU memory info
        gpu_info = self.get_gpu_memory_info()
        
        # Get system RAM info
        ram_info = psutil.virtual_memory()
        ram_used_gb = ram_info.used / 1024**3
        ram_total_gb = ram_info.total / 1024**3
        ram_percent = ram_info.percent
        
        log_entry = {
            'timestamp': timestamp,
            'stage': stage,
            'epoch': epoch,
            'ram_used_gb': ram_used_gb,
            'ram_total_gb': ram_total_gb,
            'ram_percent': ram_percent,
            'gpu_info': gpu_info,
            'additional_info': additional_info
        }
        
        self.memory_log.append(log_entry)
        
        # Print current status
        print(f"\n[{timestamp}] Memory Usage - {stage}" + (f" (Epoch {epoch})" if epoch else ""))
        print(f"RAM: {ram_used_gb:.2f}GB / {ram_total_gb:.2f}GB ({ram_percent:.1f}%)")
        
        if gpu_info:
            if 'used_mb' in gpu_info:
                print(f"GPU: {gpu_info['used_mb']:.0f}MB / {gpu_info['total_mb']:.0f}MB ({gpu_info['utilization_%']:.1f}%)")
            else:
                print(f"GPU: {gpu_info}")
        else:
            print("GPU: Not available or monitoring failed")
            
        if additional_info:
            print(f"Additional: {additional_info}")
        print("-" * 50)
    
    def plot_memory_usage(self):
        """Plot memory usage over time"""
        if not self.memory_log:
            print("No memory usage data to plot")
            return
            
        timestamps = [entry['timestamp'] for entry in self.memory_log]
        ram_usage = [entry['ram_percent'] for entry in self.memory_log]
        stages = [entry['stage'] for entry in self.memory_log]
        
        # Extract GPU usage if available
        gpu_usage = []
        for entry in self.memory_log:
            if entry['gpu_info'] and 'utilization_%' in entry['gpu_info']:
                gpu_usage.append(entry['gpu_info']['utilization_%'])
            else:
                gpu_usage.append(0)
        
        plt.figure(figsize=(15, 8))
        
        # RAM usage plot
        plt.subplot(2, 1, 1)
        plt.plot(range(len(timestamps)), ram_usage, 'b-o', linewidth=2, markersize=4)
        plt.title('RAM Usage Over Time')
        plt.ylabel('RAM Usage (%)')
        plt.grid(True, alpha=0.3)
        
        # Add stage labels
        for i, stage in enumerate(stages):
            if i % 2 == 0:  # Show every other label to avoid crowding
                plt.annotate(stage, (i, ram_usage[i]), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, rotation=45)
        
        # GPU usage plot (if available)
        plt.subplot(2, 1, 2)
        if any(usage > 0 for usage in gpu_usage):
            plt.plot(range(len(timestamps)), gpu_usage, 'r-o', linewidth=2, markersize=4)
            plt.title('GPU Memory Usage Over Time')
            plt.ylabel('GPU Memory Usage (%)')
        else:
            plt.text(0.5, 0.5, 'GPU usage data not available', 
                    transform=plt.gca().transAxes, ha='center', va='center', fontsize=14)
            plt.title('GPU Memory Usage Over Time (No Data)')
        
        plt.xlabel('Time Points')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def print_memory_summary(self):
        """Print a summary of memory usage"""
        if not self.memory_log:
            print("No memory usage data available")
            return
            
        print("\n" + "="*60)
        print("MEMORY USAGE SUMMARY")
        print("="*60)
        
        ram_usage = [entry['ram_percent'] for entry in self.memory_log]
        gpu_usage = [entry['gpu_info']['utilization_%'] for entry in self.memory_log 
                    if entry['gpu_info'] and 'utilization_%' in entry['gpu_info']]
        
        print(f"RAM Usage - Max: {max(ram_usage):.1f}%, Min: {min(ram_usage):.1f}%, Avg: {np.mean(ram_usage):.1f}%")
        
        if gpu_usage:
            print(f"GPU Usage - Max: {max(gpu_usage):.1f}%, Min: {min(gpu_usage):.1f}%, Avg: {np.mean(gpu_usage):.1f}%")
        else:
            print("GPU Usage - No data available")
            
        print(f"Total monitoring points: {len(self.memory_log)}")
        print("="*60)

class DynamicCNN:
    def compute_class_weights_tensorflow(self):
        y_train = self.train_generator.classes
        class_names = list(self.train_generator.class_indices.keys())
        class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = {i: weight for i, weight in enumerate(class_weights_arr)}
        print("Class weights (TensorFlow):")
        for i, name in enumerate(class_names):
            print(f"  {name}: {class_weights[i]:.3f}")
        return class_weights
        
    def __init__(self, num_classes=None, data_dir=DATA_DIR, train_split=TRAIN_SIZE, val_split=VAL_SIZE, test_split=TEST_SIZE):
        """
        Initialize the Dynamic CNN classifier with GPU memory monitoring

        Args:
            num_classes (int or None): Number of classes to classify. If None, will auto-detect from data
            data_dir (str): Path to the directory containing class folders
            train_split (float): Ratio for training set (default: 0.7)
            val_split (float): Ratio for validation set (default: 0.15)
            test_split (float): Ratio for testing set (default: 0.15)
        """
        self.num_classes = num_classes  # Can be None for auto-detection
        self.data_dir = data_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.model = None
        self.class_names = []
        
        # Initialize GPU memory monitor
        self.memory_monitor = GPUMemoryMonitor()
        self.memory_monitor.log_memory_usage("Initialization")

        # Validate splits
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0. Got: {train_split + val_split + test_split}")

        # Setup directories
        self.setup_directories()

    def check_system_resources(self):
        """Check GPU and RAM availability with detailed monitoring"""
        print("\n" + "="*50)
        print("SYSTEM RESOURCES CHECK")
        print("="*50)
        
        # Check GPU
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                print(f' Found {len(gpus)} GPU(s):')
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu}")
                    
                # Try to get GPU memory info
                gpu_info = self.memory_monitor.get_gpu_memory_info()
                if gpu_info and 'total_mb' in gpu_info:
                    print(f"  GPU Memory: {gpu_info['total_mb']:.0f}MB total")
                    
            else:
                print(' No GPU detected. Consider enabling GPU acceleration.')
        except Exception as e:
            print(f' Error checking GPU: {e}')

        # Check RAM
        try:
            ram_info = psutil.virtual_memory()
            ram_gb = ram_info.total / 1e9
            print(f' Available RAM: {ram_gb:.1f} GB')

            if ram_gb < 8:
                print('⚠ Warning: Low RAM detected. Consider using a high-RAM runtime.')
        except Exception as e:
            print(f' Error checking RAM: {e}')
            
        # Log initial resource state
        self.memory_monitor.log_memory_usage("Resource Check")

    def setup_directories(self):
        """Create training, validation, and testing directory structure"""
        base_dirs = [
            '/home/ubuntu/Images',
            '/home/ubuntu/Images/training',
            '/home/ubuntu/Images/validation',
            '/home/ubuntu/Images/testing'
        ]

        for dir_path in base_dirs:
            os.makedirs(dir_path, exist_ok=True)

    def extract_data(self, zip_path):
        """Extract data from zip file"""
        self.memory_monitor.log_memory_usage("Before Data Extraction")
        
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('/home/ubuntu/Images')
            print(f"Data extracted from {zip_path}")
        else:
            print(f"Zip file not found: {zip_path}")
            
        self.memory_monitor.log_memory_usage("After Data Extraction")

    def discover_classes(self, source_dir):
        """Automatically discover class names from directory structure"""
        print(f"Looking for classes in: {source_dir}")

        if os.path.exists(source_dir):
            all_items = os.listdir(source_dir)
            print(f"Found items: {all_items}")

            # Look for directories that start with 'data_'
            data_dirs = [d for d in all_items
                        if os.path.isdir(os.path.join(source_dir, d)) and d.startswith('data_')]

            # Extract class names by removing 'data_' prefix
            self.class_names = [d.replace('data_', '') for d in data_dirs]
            self.class_names.sort()  # Ensure consistent ordering

            # Auto-detect number of classes if not specified
            if self.num_classes is None:
                self.num_classes = len(self.class_names)
                print(f"Auto-detected {self.num_classes} classes")
            elif len(self.class_names) != self.num_classes:
                print(f"Warning: Found {len(self.class_names)} classes, but expected {self.num_classes}")
                print("Using the actual number found in the data")
                self.num_classes = len(self.class_names)

            print(f"Discovered classes: {self.class_names}")
            print(f"Source directories: {data_dirs}")

            # Create directories for each class
            for class_name in self.class_names:
                os.makedirs(f'/home/ubuntu/Images/training/{class_name}', exist_ok=True)
                os.makedirs(f'/home/ubuntu/Images/validation/{class_name}', exist_ok=True)
                os.makedirs(f'/home/ubuntu/Images/testing/{class_name}', exist_ok=True)
        else:
            print(f"Source directory not found: {source_dir}")

    def split_data(self, source_dir, training_dir, validation_dir, testing_dir):
        """Split data into training, validation, and testing sets"""
        if not os.path.exists(source_dir):
            print(f"Source directory does not exist: {source_dir}")
            return

        files = []
        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            if os.path.getsize(file_path) > 0:
                files.append(filename)
            else:
                print(f"{filename} is zero length, ignoring.")

        if not files:
            print(f"No valid files found in {source_dir}")
            return

        # Calculate split sizes
        total_files = len(files)
        training_length = int(total_files * self.train_split)
        validation_length = int(total_files * self.val_split)
        testing_length = total_files - training_length - validation_length  # Remaining files

        # Shuffle and split
        shuffled_set = random.sample(files, len(files))
        training_set = shuffled_set[:training_length]
        validation_set = shuffled_set[training_length:training_length + validation_length]
        testing_set = shuffled_set[training_length + validation_length:]

        # Copy training files
        for filename in training_set:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(training_dir, filename)
            copyfile(src, dst)

        # Copy validation files
        for filename in validation_set:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(validation_dir, filename)
            copyfile(src, dst)

        # Copy testing files
        for filename in testing_set:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(testing_dir, filename)
            copyfile(src, dst)

        print(f"Split {total_files} files: {len(training_set)} training, {len(validation_set)} validation, {len(testing_set)} testing")

    def prepare_data(self, source_base_dir):
        """Prepare training, validation, and testing data for all classes"""
        self.memory_monitor.log_memory_usage("Before Data Preparation")
        
        self.discover_classes(source_base_dir)

        # Map class names to their source directories
        class_to_source = {}
        for class_name in self.class_names:
            source_dir_name = f"data_{class_name}"
            class_to_source[class_name] = source_dir_name

        for class_name in self.class_names:
            source_dir_name = class_to_source[class_name]
            source_dir = os.path.join(source_base_dir, source_dir_name)
            training_dir = f'/home/ubuntu/Images/training/{class_name}'
            validation_dir = f'/home/ubuntu/Images/validation/{class_name}'
            testing_dir = f'/home/ubuntu/Images/testing/{class_name}'

            print(f"Processing class: {class_name} (from {source_dir_name})")
            self.split_data(source_dir, training_dir, validation_dir, testing_dir)
            
        self.memory_monitor.log_memory_usage("After Data Preparation")

    def visualize_sample(self):
        """Display a sample image from the dataset"""
        if self.class_names:
            sample_class = self.class_names[0]
            training_path = f'/home/ubuntu/Images/training/{sample_class}'

            if os.path.exists(training_path) and os.listdir(training_path):
                sample_file = os.listdir(training_path)[0]
                img_path = os.path.join(training_path, sample_file)

                img = image.load_img(img_path)
                plt.imshow(img)
                plt.title(f"Sample from class: {sample_class}")
                plt.axis('off')
                plt.show()

                img_array = np.array(img)
                h, w, c = img_array.shape
                print(f'Image dimensions - Width: {w}, Height: {h}, Channels: {c}')

    def build_model(self, input_shape=(150, 150, 3)):
        """Build the CNN model dynamically based on number of classes"""
        self.memory_monitor.log_memory_usage("Before Model Building")

        # Determine activation and loss function based on number of classes
        if self.num_classes == 2:
            # Binary classification
            final_activation = 'sigmoid'
            loss_function = 'binary_crossentropy'
            output_units = 1
            class_mode = 'binary'
        else:
            # Multi-class classification
            final_activation = 'softmax'
            loss_function = 'categorical_crossentropy'
            output_units = self.num_classes
            class_mode = 'categorical'

        self.class_mode = class_mode

        # Build model architecture
        self.model = tf.keras.models.Sequential([
            # First Conv Layer
            tf.keras.layers.Conv2D(32, (3,3), input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(2,2),
    
            # Second Conv Layer
            tf.keras.layers.Conv2D(64, (3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            # Third Conv Layer
            tf.keras.layers.Conv2D(128, (3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(2,2),
    
            # Dropout for regularization
            tf.keras.layers.Dropout(0.5),
    
            # Flatten and Dense Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(output_units, activation=final_activation)
            #tf.keras.layers.Flatten(),
            #tf.keras.layers.Dense(512, activation='relu'),
            #tf.keras.layers.Dense(output_units, activation=final_activation)
        ])

        # Compile model
        self.model.compile(
            loss=loss_function,
            optimizer=RMSprop(learning_rate=1e-4),
            metrics=['accuracy']
        )

        print(f"Model built for {self.num_classes} classes")
        print(f"Output activation: {final_activation}")
        print(f"Loss function: {loss_function}")
        self.model.summary()
        
        self.memory_monitor.log_memory_usage("After Model Building")

    def create_data_generators(self, batch_size=None, target_size=(150, 150)):
        """Create data generators for training and validation"""
        self.memory_monitor.log_memory_usage("Before Data Generators Creation")

        # Auto-calculate batch size if not provided
        if batch_size is None:
            # Count total training images
            total_images = 0
            training_dir = '/home/ubuntu/Images/training/'
            if os.path.exists(training_dir):
                for class_name in self.class_names:
                    class_dir = os.path.join(training_dir, class_name)
                    if os.path.exists(class_dir):
                        total_images += len(os.listdir(class_dir))

            # Calculate optimal batch size based on dataset size
            if total_images < 100:
                candidate_sizes = [4, 8, 16]
            elif total_images < 500:
                candidate_sizes = [8, 16, 32]
            elif total_images < 2000:
                candidate_sizes = [16, 32, 64]
            elif total_images < 5000:
                candidate_sizes = [32, 64, 128]
            else:
                candidate_sizes = [64, 128, 256]

            # Find the largest batch size that divides evenly (or closest to even)
            best_batch_size = candidate_sizes[0]
            smallest_remainder = total_images % candidate_sizes[0]

            for size in candidate_sizes:
                remainder = total_images % size
                if remainder == 0:  # Perfect division
                    best_batch_size = size
                    break
                elif remainder < smallest_remainder:
                    best_batch_size = size
                    smallest_remainder = remainder

            batch_size = best_batch_size
            remainder = total_images % batch_size

            print(f"Auto-selected batch size: {batch_size} (based on {total_images} training images)")
            if remainder > 0:
                print(f"Note: Last batch will have {remainder} images")

        print(f"Using batch size: {batch_size}")

        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Validation data generator (no augmentation)
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Testing data generator (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            '/home/ubuntu/Images/training/',
            batch_size=batch_size,
            class_mode=self.class_mode,
            target_size=target_size,
            shuffle=True
        )

        self.validation_generator = validation_datagen.flow_from_directory(
            '/home/ubuntu/Images/validation/',
            batch_size=batch_size,
            class_mode=self.class_mode,
            target_size=target_size,
            shuffle=False
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            '/home/ubuntu/Images/testing/',
            batch_size=batch_size,
            class_mode=self.class_mode,
            target_size=target_size,
            shuffle=False
        )

        print(f"Found {self.train_generator.samples} training images")
        print(f"Found {self.validation_generator.samples} validation images")
        print(f"Found {self.test_generator.samples} testing images")
        print(f"Class indices: {self.train_generator.class_indices}")
        
        self.memory_monitor.log_memory_usage("After Data Generators Creation", 
                                           additional_info=f"Batch size: {batch_size}")

    def train(self, epochs=EPOCH, steps_per_epoch=None, validation_steps=None):
        if self.model is None:
            print("Model not built. Call build_model() first.")
            return None

        if not hasattr(self, 'train_generator'):
            print("Data generators not created. Call create_data_generators() first.")
            return None

    
        if steps_per_epoch is None:
            steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size

        if validation_steps is None:
            validation_steps = self.validation_generator.samples // self.validation_generator.batch_size

    # === Compute Class Weights Based on Training Set Frequencies ===
        print("\n" + "="*50)
        print("CLASS WEIGHTING FOR IMBALANCED DATA")
        print("="*50)

        y_train = self.train_generator.classes
        class_indices = self.train_generator.class_indices
        class_names = [name for name, _ in sorted(class_indices.items(), key=lambda x: x[1])]  # Sorted by index

        
        class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = {i: weight for i, weight in enumerate(class_weights_arr)}

        print("Class distribution in training set:")
        for class_name in class_names:
            dir_path = os.path.join(self.data_dir, 'training', class_name)
            count = len(os.listdir(dir_path)) if os.path.exists(dir_path) else 0
            print(f"  {class_name}: {count} samples")

        print("\nApplied class weights (inverse of frequency):")
        for i, name in enumerate(class_names):
            print(f"  {name}: {class_weights[i]:.3f}")

        # Log memory before training
        self.memory_monitor.log_memory_usage("Before Training Start")

    # Custom callback for epoch-wise memory monitoring

        class MemoryCallback(tf.keras.callbacks.Callback):
            def __init__(self, memory_monitor):
                self.memory_monitor = memory_monitor
            
            def on_epoch_begin(self, epoch, logs=None):
                self.memory_monitor.log_memory_usage("Training", epoch + 1)
            
            def on_epoch_end(self, epoch, logs=None):
                self.memory_monitor.log_memory_usage("Training End", epoch + 1, 
                                               additional_info=f"Loss: {logs.get('loss', 'N/A'):.4f}, "
                                                              f"Acc: {logs.get('accuracy', 'N/A'):.4f}, "
                                                              f"Val_Acc: {logs.get('val_accuracy', 'N/A'):.4f}")
        memory_callback = MemoryCallback(self.memory_monitor)

    # Train the model with class weighting
        history = self.model.fit(self.train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=self.validation_generator, validation_steps=validation_steps, callbacks=[memory_callback], class_weight=class_weights, verbose=1)

    # Log memory after training
        self.memory_monitor.log_memory_usage("After Training Complete")
        return history

    def evaluate_model(self):
        """Evaluate the model using the TEST set with memory monitoring"""
        if self.model is None:
            print("Model not trained. Call train() first.")
            return

        if not hasattr(self, 'test_generator'):
            print("Test generator not created. Call create_data_generators() first.")
            return

        print("\n" + "="*50)
        print("MODEL EVALUATION ON TEST SET")
        print("="*50)

        # Log memory before evaluation
        self.memory_monitor.log_memory_usage("Before Testing")

        # Reset the test generator
        self.test_generator.reset()

        # Get predictions on TEST set
        print("Generating predictions on test set...")
        predictions = self.model.predict(self.test_generator, verbose=1)

        # Log memory after predictions
        self.memory_monitor.log_memory_usage("After Predictions Generated")

        # Get true labels from TEST set
        true_labels = self.test_generator.classes

        # Convert predictions based on classification type
        if self.class_mode == 'binary':
            # Binary classification
            predicted_labels = (predictions > 0.5).astype(int).flatten()
            class_labels = list(self.test_generator.class_indices.keys())
        else:
            # Multi-class classification
            predicted_labels = np.argmax(predictions, axis=1)
            # Create ordered class labels based on class_indices
            class_indices = self.test_generator.class_indices
            class_labels = [None] * len(class_indices)
            for class_name, index in class_indices.items():
                class_labels[index] = class_name

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Calculate precision, recall, and f1-score with appropriate averaging
        if self.num_classes == 2:
            precision = precision_score(true_labels, predicted_labels, average='binary')
            recall = recall_score(true_labels, predicted_labels, average='binary')
            f1 = f1_score(true_labels, predicted_labels, average='binary')
        else:
            precision = precision_score(true_labels, predicted_labels, average='weighted')
            recall = recall_score(true_labels, predicted_labels, average='weighted')
            f1 = f1_score(true_labels, predicted_labels, average='weighted')

        # Print overall metrics
        print("\nTEST SET METRICS:")
        print("-" * 30)
        print(f"Test Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test F1-Score:  {f1:.4f}")
        print(f"Test samples:   {len(true_labels)}")

        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

        # Print detailed classification report
        print("\nDETAILED CLASSIFICATION REPORT (TEST SET):")
        print("-" * 50)
        report = classification_report(true_labels, predicted_labels, 
                                     target_names=class_labels, digits=4)
        print(report)

        # Print confusion matrix in text format
        print("\nCONFUSION MATRIX (TEST SET):")
        print("-" * 30)
        print("Rows: True Labels, Columns: Predicted Labels")
        print()
        
        # Print header
        header = "True\\Pred"
        for label in class_labels:
            header += f"\t{label[:8]}"  # Truncate long labels
        print(header)
        
        # Print matrix with row labels
        for i, true_label in enumerate(class_labels):
            row = f"{true_label[:8]}"  # Truncate long labels
            for j in range(len(class_labels)):
                row += f"\t{cm[i, j]}"
            print(row)

        # Calculate and print per-class metrics
        print("\nPER-CLASS METRICS (TEST SET):")
        print("-" * 35)
        for i, class_name in enumerate(class_labels):
            if self.num_classes == 2:
                # For binary classification, calculate metrics for each class
                class_precision = precision_score(true_labels, predicted_labels, 
                                                labels=[i], average=None)
                class_recall = recall_score(true_labels, predicted_labels, 
                                          labels=[i], average=None)
                class_f1 = f1_score(true_labels, predicted_labels, 
                                   labels=[i], average=None)
                
                if len(class_precision) > 0:
                    print(f"{class_name}: Precision={class_precision[0]:.4f}, "
                          f"Recall={class_recall[0]:.4f}, F1={class_f1[0]:.4f}")
            else:
                # For multi-class, extract from classification report
                pass  # The detailed report above already shows per-class metrics

        # Log memory after evaluation complete
        self.memory_monitor.log_memory_usage("After Testing Complete")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'class_labels': class_labels
        }

    def plot_training_history(self, history):
        """Plot training history"""
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.show()

    def plot_memory_usage(self):
        """Plot memory usage throughout the training process"""
        self.memory_monitor.plot_memory_usage()
        
    def print_memory_summary(self):
        """Print memory usage summary"""
        self.memory_monitor.print_memory_summary()

# Example usage - FULLY AUTOMATIC class detection with GPU memory monitoring:
def main():
    print("="*60)
    print("DYNAMIC CNN WITH GPU MEMORY MONITORING")
    print("="*60)


    classifier = DynamicCNN(
        num_classes=None, 
        data_dir=DATA_DIR,
        train_split=TRAIN_SIZE,
        val_split=VAL_SIZE, 
        test_split=TEST_SIZE
    )

    # Check system resources
    classifier.check_system_resources()

    # Extract data (uncomment if you need to extract the zip file)
    classifier.extract_data(ZIP_PATH)

    # Prepare data - automatically detects ALL data_* directories and splits into 3 sets
    classifier.prepare_data(DATA_DIR)

    # The code will automatically report how many classes it found
    print(f"Automatically detected {classifier.num_classes} classes: {classifier.class_names}")
    print(f"Data split: {classifier.train_split*100:.0f}% train, {classifier.val_split*100:.0f}% validation, {classifier.test_split*100:.0f}% test")

    # Visualize a sample
    classifier.visualize_sample()

    # Build model - automatically configures for the detected number of classes
    classifier.build_model()

    # Create data generators - automatically optimizes batch size
    classifier.create_data_generators()

    # Train model using training set and validate on validation set (with memory monitoring)
    history = classifier.train(epochs=EPOCH)

    # Plot training results
    if history:
        classifier.plot_training_history(history)

    # Evaluate model on TEST set and show confusion matrix with all metrics
    evaluation_results = classifier.evaluate_model()
    
    # Plot memory usage throughout the process
    print("\n" + "="*60)
    print("MEMORY USAGE ANALYSIS")
    print("="*60)
    classifier.plot_memory_usage()
    classifier.print_memory_summary()
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - FINAL TEST RESULTS")
    print("="*60)
    if evaluation_results:
        print(f"Final Test Accuracy: {evaluation_results['accuracy']:.4f} ({evaluation_results['accuracy']*100:.2f}%)")
        print(f"Final Test Precision: {evaluation_results['precision']:.4f}")
        print(f"Final Test Recall: {evaluation_results['recall']:.4f}")
        print(f"Final Test F1-Score: {evaluation_results['f1_score']:.4f}")
        print(f"Test Set Size: {len(evaluation_results['true_labels'])} samples")

# Enhanced examples for specific scenarios:
def binary_classification_with_monitoring():
    """Example for binary classification (2 classes) with GPU monitoring"""
    print("Binary Classification with GPU Memory Monitoring")
    print("="*50)
    
    # If you have data_class1/ and data_class2/
    classifier = DynamicCNN(num_classes=2, data_dir="/home/ubuntu/Images/")
    
    # All the monitoring happens automatically
    classifier.check_system_resources()
    classifier.prepare_data("/home/ubuntu/Images/")
    classifier.build_model()
    classifier.create_data_generators()
    history = classifier.train(epochs=EPOCH)
    
    if history:
        classifier.plot_training_history(history)
    
    classifier.evaluate_model()
    classifier.plot_memory_usage()
    classifier.print_memory_summary()

def detailed_memory_monitoring_example():
    """Example with detailed memory monitoring at custom intervals"""
    print("Detailed Memory Monitoring Example")
    print("="*50)
    
    classifier = DynamicCNN(num_classes=None, data_dir=DATA_DIR)
    
    # Manual memory logging at specific points
    classifier.memory_monitor.log_memory_usage("Custom Checkpoint 1")
    
    classifier.check_system_resources()
    classifier.prepare_data(DATA_DIR)
    
    classifier.memory_monitor.log_memory_usage("Custom Checkpoint 2", additional_info="After data preparation")
    
    classifier.build_model()
    
   
    classifier.create_data_generators()  
    
    classifier.memory_monitor.log_memory_usage("Custom Checkpoint 3", additional_info=f"Before training with auto batch_size")

   
    history = classifier.train(epochs=EPOCH)  
    
    if history:
        classifier.plot_training_history(history)
    
    classifier.evaluate_model()
    
    # Show detailed memory analysis
    print("\n" + "="*60)
    print("MEMORY USAGE ANALYSIS")
    print("="*60)
    classifier.plot_memory_usage()
    classifier.print_memory_summary()

# Installation requirements note
def print_requirements():
    """Print additional package requirements for GPU monitoring"""
    print("\n" + "="*60)
    print("ADDITIONAL REQUIREMENTS FOR FULL GPU MONITORING")
    print("="*60)
    print("For detailed GPU memory monitoring, install:")
    print("pip install pynvml")
    print("pip install psutil")
    print("")
    print("pynvml: NVIDIA GPU monitoring")
    print("psutil: System resource monitoring")
    print("="*60)

if __name__ == "__main__":
    print_requirements()
    main()