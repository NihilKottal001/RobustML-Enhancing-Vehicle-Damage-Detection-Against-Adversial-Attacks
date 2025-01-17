import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.defences.preprocessor import GaussianAugmentation




class AdversarialUtils:
    def __init__(self, classifier, dataset_categories):
        """
        Initialize the utility class.
        
        Args:
            classifier: An ART PyTorchClassifier instance.
            dataset_categories: A list of category names from the dataset.
        """
        self.classifier = classifier
        self.dataset_categories = dataset_categories

    def filter_by_label_name(self, loader, label_name, dataset):
        """
        Filters the DataLoader by a specific label name.

        Args:
            loader: DataLoader to filter.
            label_name: The label name to filter by.
            dataset: Dataset instance with category names.

        Returns:
            A DataLoader containing only the filtered images and labels.
        """
        filtered_images = []
        filtered_labels = []

        for images, labels in loader:
            for i, label in enumerate(labels):
                if dataset.categories[label] == label_name:
                    filtered_images.append(images[i])
                    filtered_labels.append(labels[i])

        filtered_images = torch.stack(filtered_images)
        filtered_labels = torch.tensor(filtered_labels)
        filtered_dataset = TensorDataset(filtered_images, filtered_labels)
        return DataLoader(filtered_dataset, batch_size=4, shuffle=False)

    @staticmethod
    def imshow(img, title=None):
        """
        Display an image tensor using matplotlib.

        Args:
            img: Tensor image to display.
            title: Title for the image.
        """
        img = img / 2 + 0.5  # Adjust for normalization
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if title:
            plt.title(title)
        plt.axis('off')

    def generate_adversarial_examples(self, data, epsilon=0.1):
        """
        Generates adversarial examples using the Fast Gradient Method.

        Args:
            data: Input data (numpy array) to generate adversarial examples for.
            epsilon: The perturbation intensity.

        Returns:
            Adversarial examples (numpy array).
        """
        attack = FastGradientMethod(estimator=self.classifier, eps=epsilon)
        return attack.generate(x=data)

    def generate_adv_loader(self, filtered_loader, batch_size=4, epsilon=0.1):
        """
        Generate a DataLoader with adversarial examples for the entire filtered loader.

        Args:
            filtered_loader (DataLoader): The filtered DataLoader containing clean images.
            batch_size (int): Batch size for the new adversarial DataLoader.
            epsilon (float): Perturbation intensity for the adversarial attack.

        Returns:
            DataLoader: A new DataLoader containing the adversarial examples and their labels.
        """
        attack = FastGradientMethod(estimator=self.classifier, eps=epsilon)

        all_adv_images = []
        all_labels = []

        # Iterate through the filtered DataLoader
        for images, labels in filtered_loader:
            # Convert images to NumPy array for ART compatibility
            images_np = images.numpy()
            labels_np = labels.numpy()

            # Generate adversarial examples
            adv_images_np = attack.generate(x=images_np)

            # Collect the adversarial examples and their labels
            all_adv_images.append(torch.tensor(adv_images_np, dtype=torch.float32))
            all_labels.append(torch.tensor(labels_np, dtype=torch.long))

        # Concatenate all adversarial images and labels into single tensors
        all_adv_images = torch.cat(all_adv_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Create a TensorDataset and DataLoader
        adv_dataset = TensorDataset(all_adv_images, all_labels)
        adv_loader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=False)

        return adv_loader


    def generate_transformed_loader(self, input_loader, transform_func, batch_size=4):
        """
        Generate a DataLoader by applying a transformation to all images in the input DataLoader.

        Args:
            input_loader (DataLoader): The DataLoader containing the original images and labels.
            transform_func (callable): A function to apply to the images (e.g., adversarial attack or defense).
            batch_size (int): Batch size for the new DataLoader.

        Returns:
            DataLoader: A new DataLoader containing the transformed images and their labels.
        """
        transformed_images = []
        transformed_labels = []

        # Iterate through the input DataLoader
        for images, labels in input_loader:
            # Convert images to NumPy for compatibility with transform_func
            images_np = images.numpy()

            # Apply the transformation function
            transformed_images_np = transform_func(images_np)

            # Convert back to tensors and collect
            transformed_images.append(torch.tensor(transformed_images_np, dtype=torch.float32))
            transformed_labels.append(labels)  # Labels remain unchanged

        # Concatenate all transformed images and labels into single tensors
        transformed_images = torch.cat(transformed_images, dim=0)
        transformed_labels = torch.cat(transformed_labels, dim=0)

        # Create a TensorDataset and DataLoader
        transformed_dataset = TensorDataset(transformed_images, transformed_labels)
        transformed_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False)

        return transformed_loader


    @staticmethod
    def nonlinear_transform(x):
        """
        Apply a nonlinear transformation to data for defense.

        Args:
            x: Input data (numpy array) to transform.

        Returns:
            Transformed data (numpy array).
        """
        x = np.exp(5 * x)
        x[x > 200] = 255
        return x


    @staticmethod
    def median_filter(x, kernel_size=3):
        """
        Apply median filtering to data for defense.

        Args:
            x: Input data (numpy array) of shape (N, C, H, W).
            kernel_size: Size of the median filter kernel.

        Returns:
            Transformed data (numpy array) with median filtering applied.
        """
        import cv2

        filtered_images = []
        for img in x:
            img = cv2.medianBlur(img.transpose(1, 2, 0).astype(np.uint8), kernel_size)  # Convert to HWC for OpenCV
            filtered_images.append(img.transpose(2, 0, 1))  # Convert back to CHW
        return np.stack(filtered_images)
    
    
    @staticmethod
    def jpeg_compression(x, quality=75):
        """
        Apply JPEG compression to data for defense.

        Args:
            x: Input data (numpy array) of shape (N, C, H, W).
            quality: Quality of the JPEG compression (1 to 100, higher is better).

        Returns:
            Transformed data (numpy array) with JPEG compression applied.
        """
        from PIL import Image
        import io

        compressed_images = []
        for img in x:
            # Convert to PIL Image format
            img = Image.fromarray((img * 255).astype(np.uint8).transpose(1, 2, 0))  # Convert CHW -> HWC

            # Apply JPEG compression
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            img = Image.open(buffer)

            # Convert back to NumPy array (CHW format)
            compressed_images.append(np.array(img).transpose(2, 0, 1) / 255.0)  # Convert HWC -> CHW

        return np.stack(compressed_images)
    
    @staticmethod
    def normalize(x, mean=0.5, std=0.5):
        return (x - mean) / std
    

    @staticmethod
    def gaussian_blur(x, kernel_size=3):
        """
        Apply Gaussian blur to data for defense.

        Args:
            x: Input data (numpy array) of shape (N, C, H, W).
            kernel_size: Size of the Gaussian kernel.

        Returns:
            Transformed data (numpy array) with Gaussian blur applied.
        """
        import cv2

        blurred_images = []
        for img in x:
            # Convert CHW -> HWC for OpenCV
            img = cv2.GaussianBlur(img.transpose(1, 2, 0), (kernel_size, kernel_size), 0)
            # Convert back to CHW
            blurred_images.append(img.transpose(2, 0, 1))
        return np.stack(blurred_images)
    

    def adaptive_gaussian_blur(self, x, max_kernel_size=7):
        """
        Apply Gaussian blur with adaptive kernel size based on the input's variance.

        Args:
            x: Input data (numpy array) of shape (N, C, H, W).
            max_kernel_size: Maximum kernel size for the blur.

        Returns:
            Transformed data (numpy array) with adaptive Gaussian blur applied.
        """
        import cv2
        blurred_images = []
        for img in x:
            variance = np.var(img)  # Compute variance as a proxy for noise level
            kernel_size = int(max(3, min(max_kernel_size, variance * 10)))  # Map variance to kernel size
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure odd kernel size
            img = cv2.GaussianBlur(img.transpose(1, 2, 0), (kernel_size, kernel_size), 0)  # HWC format for OpenCV
            blurred_images.append(img.transpose(2, 0, 1))  # Convert back to CHW
        return np.stack(blurred_images)


    def combined_gaussian_jpeg(self, x, kernel_size=3, jpeg_quality=75):
        """
        Apply Gaussian blur followed by JPEG compression to data for defense.

        Args:
            x: Input data (numpy array) of shape (N, C, H, W).
            kernel_size: Size of the Gaussian kernel.
            jpeg_quality: Quality of the JPEG compression (1 to 100, higher is better).

        Returns:
            Transformed data (numpy array) with combined transformations applied.
        """
        # Apply Gaussian Blur
        #x = self.gaussian_blur(x, kernel_size=kernel_size)
        x = self.adaptive_gaussian_blur(x, max_kernel_size=kernel_size)

        # Apply JPEG Compression
        x = self.jpeg_compression(x, quality=jpeg_quality)

        return x
    
    def bit_depth_reduction(self, x, bit_depth=4):
        """
        Apply bit depth reduction to the input data.

        Args:
            x: Input data (numpy array) to reduce bit depth.
            bit_depth: Number of bits to keep (e.g., 4 for 16 levels).

        Returns:
            Transformed data with reduced bit depth.
        """
        max_val = 2 ** bit_depth - 1
        return np.round(x * max_val) / max_val
    
    def combined_adaptive_gaussian_jpeg_bitdepth(self, x, kernel_size=7, jpeg_quality=50, bit_depth=4):
        x = self.combined_gaussian_jpeg(x, kernel_size=kernel_size, jpeg_quality=jpeg_quality)
        x = self.bit_depth_reduction(x, bit_depth=bit_depth)
        x = x + np.random.normal(0, 0.01, x.shape)  # Add Gaussian noise

        # Average the results of all transformations
       
        return x


    def apply_gaussian_noise(self, x, std=0.1):
        """
        Apply Gaussian noise to data using ART's GaussianNoise defense.

        Args:
            x: Input data (numpy array) of shape (N, C, H, W).
            mean: Mean of the Gaussian noise.
            std: Standard deviation of the Gaussian noise.

        Returns:
            Transformed data with Gaussian noise applied (numpy array).
        """
        defense = GaussianAugmentation(sigma=std)
        smoothed_images, _ = defense(x=x)  # Apply noise
        return smoothed_images

