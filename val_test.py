import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from models import load_model
from PIL import Image
import pandas as pd
from pathlib import Path

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

class ValidationDataset(Dataset):
    def __init__(self, val_dir, csv_file, transform=None):
        self.val_dir = Path(val_dir)
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        label_str = self.labels_df.iloc[idx, 1]
        label = CLASS_TO_IDX[label_str]  # Convert class name to integer

        img_path = self.val_dir / img_name
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

def test_model(model_path, val_dir, csv_file, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    # model = load_model("classifier", with_weights=True)
    model_path = "classification/classifier.th"
    model = load_model("classifier", with_weights=True, model_path = model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    val_dataset = ValidationDataset(val_dir=val_dir, csv_file=csv_file, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            pred = model.predict(img)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_pred_labels = [IDX_TO_CLASS[pred] for pred in all_preds]
    all_true_labels = [IDX_TO_CLASS[label] for label in all_labels]

    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    precision = precision_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
    recall = recall_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
    f1 = f1_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)

    class_report = classification_report(all_true_labels, all_pred_labels, target_names=CLASS_NAMES, zero_division=0)

    false_positives = {class_name: 0 for class_name in CLASS_NAMES}
    false_negatives = {class_name: 0 for class_name in CLASS_NAMES}

    for true_label, pred_label in zip(all_true_labels, all_pred_labels):
        if true_label != pred_label:
            false_positives[pred_label] += 1  
            false_negatives[true_label] += 1  

    # Display results
    print(f"Accuracy: {accuracy:.4f}")
    print("Per-Class Metrics:\n")
    print(class_report)

    print("\nFalse Positives and False Negatives:")
    for class_name in CLASS_NAMES:
        print(f"{class_name}: False Positives = {false_positives[class_name]}, False Negatives = {false_negatives[class_name]}")

    return accuracy, precision, recall, f1, false_positives, false_negatives


if __name__ == "__main__":
    model_path = "classification/classifier.th"  # Path to the trained model file
    val_dir = "classification_data/val"  # Path to the validation images
    csv_file = "classification_data/val/labels.csv"  # Path to the CSV file with image names and labels

    test_model(model_path, val_dir, csv_file)
