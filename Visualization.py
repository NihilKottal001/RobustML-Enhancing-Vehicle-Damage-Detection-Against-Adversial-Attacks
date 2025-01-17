import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class LIMEExplainer:
    def __init__(self, model, device, data_loader):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.explainer = lime_image.LimeImageExplainer()

    def batch_predict(self, images):
        # Prepare images as expected by the model
        images = torch.stack([TF.to_tensor(TF.to_pil_image(img)) for img in images]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
        return torch.softmax(outputs, dim=1).cpu().numpy()

    def get_explainability_score(self, image, label, num_samples=1000):
        explanation = self.explainer.explain_instance(
            image,
            self.batch_predict,  # Use the prediction function defined
            top_labels=5,
            hide_color=0,
            num_samples=num_samples
        )
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=False)
        explainability_score = np.mean(mask)
        return explainability_score, top_label

    def visualize_explanation(self, image, label):
        # Ensure the image is in CPU and convert to PIL for processing
        test_image_for_lime = TF.to_pil_image(image.cpu())

        # Convert the PIL Image to NumPy array for LIME
        test_image_np = np.array(test_image_for_lime)

        explanation = self.explainer.explain_instance(
            test_image_np,
            self.batch_predict,
            top_labels=1,
            hide_color=0,  # This sets the background color of non-explanation areas
            num_samples=1000  # Number of perturbations to use for explanation
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False  # Shows the image with only the explanation mask applied
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(mark_boundaries(temp, mask))
        plt.title('Highlighted Features by LIME')
        plt.axis('off')
        plt.show()