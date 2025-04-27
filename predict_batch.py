import torch
import random
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from models import load_model
from pathlib import Path
import csv

# Constants
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
GRID_ROWS, GRID_COLS = 7,7
CELL_WIDTH, CELL_HEIGHT = 224, 224
BORDER_WIDTH = 4
FONT_SIZE = 16

def load_labels(labels_csv):
    label_dict = {}
    with open(labels_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for fname, label in reader:
            label_dict[fname] = label
    return label_dict

def predict_single(model, image_path, transform, device):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_idx = model.predict(input_tensor).item()
        pred_label = CLASS_NAMES[pred_idx]
    return pred_label, img.resize((CELL_WIDTH, CELL_HEIGHT))

def draw_text_and_border(img, pred_label, true_label, is_correct):
    border_color = "green" if is_correct else "red"
    draw = ImageDraw.Draw(img)
    verdict = "Correct" if is_correct else "Incorrect"
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
    draw.text((5, 5), f"P: {pred_label}", fill="red", font=font)
    draw.text((5, 25), f"T: {true_label}", fill="blue", font=font)
    draw.text((5, 45), verdict, fill=border_color, font=font)

    # Draw border around the image
    for i in range(BORDER_WIDTH):
        draw.rectangle(
            [i, i, img.width - 1 - i, img.height - 1 - i],
            outline=border_color
        )
    return img

def create_image_grid(results, rows, cols):
    grid_img = Image.new("RGB", (cols * CELL_WIDTH, rows * CELL_HEIGHT), "white")

    for idx, (img, pred, true, is_correct) in enumerate(results):
        r, c = divmod(idx, cols)
        annotated = draw_text_and_border(img, pred, true, is_correct)
        grid_img.paste(annotated, (c * CELL_WIDTH, r * CELL_HEIGHT))

    return grid_img

def main():
    val_dir = Path("classification_data/val")
    labels_csv = val_dir / "labels.csv"
    model_path = "classification/classifier.th"

    label_dict = load_labels(labels_csv)
    image_files = list(val_dir.glob("*.jpg"))
    selected_images = random.sample(image_files, GRID_ROWS * GRID_COLS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("classifier", with_weights=True, model_path=model_path).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    results = []
    correct = 0

    for image_path in selected_images:
        true_label = label_dict.get(image_path.name, "UNKNOWN")
        pred_label, resized_img = predict_single(model, image_path, transform, device)
        is_correct = pred_label == true_label
        results.append((resized_img, pred_label, true_label, is_correct))
        if is_correct:
            correct += 1

    final_grid = create_image_grid(results, GRID_ROWS, GRID_COLS)
    final_grid.save("predictions_grid.png")
    final_grid.show()

    print(f"\nAccuracy: {correct} / {GRID_ROWS * GRID_COLS} = {correct / (GRID_ROWS * GRID_COLS):.2%}")

if __name__ == "__main__":
    main()
