# UI code

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

def load_model(model_path, num_classes, device):
    from torchvision.models import mobilenet_v3_small
    import torch.nn as nn
    model = mobilenet_v3_small(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_calorie_dict(csv_path):
    df = pd.read_csv(csv_path)
    calorie_dict = dict(zip(df['Class'], df['Calories per 100g']))
    class_names = list(df['Class'])
    return calorie_dict, class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"C:\Users\kaush\Documents\ASE-ECE\Sem 5\AIML\FoodSeg103 Stuff\Kaush Stuff\final_image_classifier.pth"

csv_path = st.text_input(r"C:\Users\kaush\Documents\ASE-ECE\Sem 5\AIML\FoodSeg103 Stuff\Kaush Stuff\nutrition_database.csv")

if csv_path:
    calorie_dict, class_names = load_calorie_dict(csv_path)
    num_classes = len(class_names)
    model = load_model(model_path, num_classes, device)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def predict(model, img, device):
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = torch.sigmoid(model(img))
            preds = (outputs > 0.5)[0].cpu().numpy()
        return preds

    st.title("Food Calorie Estimator")
    image_file = st.file_uploader("Upload food image", type=["jpg", "png", "jpeg"])
    weight = st.number_input("Enter food weight (g)", min_value=1, value=100)

    if image_file and weight:
        img = Image.open(image_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        preds = predict(model, img, device)
        detected_foods = [class_names[i] for i, p in enumerate(preds) if p and class_names[i].lower() != "background"]
        total_cal = 0
        cal_details = {}
        for food in detected_foods:
            cal = calorie_dict.get(food, 0) * weight / 100
            cal_details[food] = cal
            total_cal += cal
        st.write("#### Predicted Foods:")
        st.write(", ".join(detected_foods) if detected_foods else "None")
        st.write(f"#### Total Estimated Calories for {weight}g:")
        st.write(f"{total_cal:.2f} kcal")
        st.write("#### Breakdown:")
        for food, val in cal_details.items():
            st.write(f"{food}: {val:.2f} kcal")
else:
    st.info("Awaiting CSV nutrition file path…")