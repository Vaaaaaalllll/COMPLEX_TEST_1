# WARNING: template code, may need edits
import torch
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
from torchvision import transforms

from config import Config
from model import load_model

class Inferencer:
    """Inferencer class for making predictions on new images"""
    
    def __init__(self, model_path=Config.BEST_MODEL_PATH):
        Config.create_dirs()
        
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Define transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Load and preprocess an image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image, image_tensor
    
    def predict(self, image_path, visualize=True):
        """Make prediction on a single image"""
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
        
        # Load and preprocess image
        original_image, image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(Config.DEVICE)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        # Get results
        predicted_class = Config.CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()
        
        # Print results
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Image: {image_path}")
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence_score*100:.2f}%")
        print("="*50)
        
        # Show probabilities for all classes
        print("\nClass Probabilities:")
        for idx, class_name in enumerate(Config.CLASS_NAMES):
            prob = probabilities[0][idx].item()
            print(f"  {class_name}: {prob*100:.2f}%")
        
        # Visualize if requested
        if visualize:
            self.visualize_prediction(
                original_image, 
                predicted_class, 
                confidence_score,
                probabilities[0].cpu().numpy(),
                image_path
            )
        
        return predicted_class, confidence_score
    
    def visualize_prediction(self, image, predicted_class, confidence, probabilities, image_path):
        """Visualize the prediction with the original image"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f"Prediction: {predicted_class}\nConfidence: {confidence*100:.1f}%", 
                     fontsize=12, fontweight='bold')
        
        # Display probability bar chart
        colors = ['green' if i == probabilities.argmax() else 'gray' 
                 for i in range(len(Config.CLASS_NAMES))]
        ax2.barh(Config.CLASS_NAMES, probabilities * 100, color=colors)
        ax2.set_xlabel('Confidence (%)', fontsize=10)
        ax2.set_title('Class Probabilities', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        
        # Add percentage labels
        for i, prob in enumerate(probabilities):
            ax2.text(prob * 100 + 2, i, f'{prob*100:.1f}%', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save visualization
        output_filename = os.path.basename(image_path).rsplit('.', 1)[0] + '_prediction.png'
        output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {output_path}")
        
        plt.close()
    
    def predict_batch(self, image_folder):
        """Make predictions on all images in a folder"""
        if not os.path.exists(image_folder):
            print(f"Error: Folder not found at {image_folder}")
            return
        
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No images found in {image_folder}")
            return
        
        print(f"\nFound {len(image_files)} images. Processing...\n")
        
        results = []
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            predicted_class, confidence = self.predict(img_path, visualize=False)
            results.append((img_file, predicted_class, confidence))
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        for img_file, pred_class, conf in results:
            print(f"{img_file:30s} -> {pred_class:10s} ({conf*100:.1f}%)")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Cat Image Classifier - Inference')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--folder', type=str, help='Path to a folder of images')
    parser.add_argument('--model', type=str, default=Config.BEST_MODEL_PATH,
                       help='Path to model file')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Create inferencer
    inferencer = Inferencer(model_path=args.model)
    
    # Run inference
    if args.image:
        inferencer.predict(args.image, visualize=not args.no_viz)
    elif args.folder:
        inferencer.predict_batch(args.folder)
    else:
        print("Please provide either --image or --folder argument")
        parser.print_help()

if __name__ == "__main__":
    main()
