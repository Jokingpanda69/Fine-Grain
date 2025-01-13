import os
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import numpy as np

class DatasetAnalyzer:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(dataset_dir, 'train')
        self.val_dir = os.path.join(dataset_dir, 'val')
        self.test_dir = os.path.join(dataset_dir, 'test')
        self.data_stats = {}

    def analyze_data_distribution(self):
        """Analyze the number of images per class in train, val, and test datasets."""
        for data_type, data_path in [('Train', self.train_dir), ('Validation', self.val_dir), ('Test', self.test_dir)]:
            if not os.path.exists(data_path):
                print(f"{data_type} directory not found at {data_path}")
                continue
            class_counts = Counter()
            for class_name in os.listdir(data_path):
                class_path = os.path.join(data_path, class_name)
                if os.path.isdir(class_path):
                    class_counts[class_name] += len(os.listdir(class_path))
            self.data_stats[data_type] = class_counts
            print(f"\n{data_type} Dataset:")
            for cls, count in class_counts.items():
                print(f"  {cls}: {count} images")
        return self.data_stats

    def visualize_sample_images(self, num_samples=5):
        """Display sample images from each class in the training dataset."""
        if not os.path.exists(self.train_dir):
            print(f"Training directory not found at {self.train_dir}")
            return
        print("\nVisualizing sample images from the training dataset...")
        for class_name in os.listdir(self.train_dir):
            class_path = os.path.join(self.train_dir, class_name)
            if os.path.isdir(class_path):
                sample_images = os.listdir(class_path)[:num_samples]
                plt.figure(figsize=(15, 5))
                plt.suptitle(f"Class: {class_name}", fontsize=16)
                for i, img_name in enumerate(sample_images):
                    img_path = os.path.join(class_path, img_name)
                    img = Image.open(img_path)
                    plt.subplot(1, num_samples, i + 1)
                    plt.imshow(img)
                    plt.axis('off')
                plt.show()

    def analyze_image_dimensions(self):
        """Analyze the dimensions of images in the dataset."""
        print("\nAnalyzing image dimensions...")
        dims = []
        for data_type, data_path in [('Train', self.train_dir), ('Validation', self.val_dir), ('Test', self.test_dir)]:
            if not os.path.exists(data_path):
                continue
            for class_name in os.listdir(data_path):
                class_path = os.path.join(data_path, class_name)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            img = Image.open(img_path)
                            dims.append(img.size)
                        except Exception as e:
                            print(f"Error reading {img_path}: {e}")
        if dims:
            dims_array = np.array(dims)
            print(f"Image dimensions (HxW): min={dims_array.min(axis=0)}, max={dims_array.max(axis=0)}, mean={dims_array.mean(axis=0)}")
        else:
            print("No images found to analyze.")

    def analyze(self):
        """Perform all analyses."""
        self.analyze_data_distribution()
        self.visualize_sample_images()
        self.analyze_image_dimensions()

if __name__ == "__main__":
    dataset_dir = r"C:\Users\bhave\PycharmProjects\HMKB\data"
    analyzer = DatasetAnalyzer(dataset_dir)
    analyzer.analyze()
