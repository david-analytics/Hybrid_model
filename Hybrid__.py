import streamlit as st
import os
import numpy as np
import pandas as pd
import pickle


# Define Capsule Network class
class CapsuleNetwork(nn.Module):
    def __init__(self, n_classes, n_routing=3):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.1)
        self.primary_caps = nn.Conv2d(in_channels=256, out_channels=32 * 8, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc_caps = nn.Linear(32 * 8 * 8, n_classes * 16)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc_caps_out = nn.Linear(n_classes * 16, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.dropout2(x)
        x = F.relu(self.primary_caps(x))
        x = self.flatten(x)
        x = F.relu(self.fc_caps(x))
        x = self.fc_dropout(x)
        x = self.fc_caps_out(x)
        return x

# Define Hybrid Model class
class HybridModel:
    def __init__(self, capsule_model, rf_model):
        self.capsule_model = capsule_model.to(device)
        self.rf_model = rf_model

    def predict(self, input_type, input_data):
        if input_type == "image":
            with torch.no_grad():
                input_data = input_data.to(device)
                output = self.capsule_model(input_data)
                return F.softmax(output, dim=1).cpu().numpy()
        elif input_type == "csv":
            return self.rf_model.predict_proba(input_data)
        else:
            raise ValueError("Invalid input_type. Must be 'image' or 'csv'.")

# Load Capsule Network and Random Forest models
@st.cache_resource
def load_models():
    n_classes = 25
    caps_model = CapsuleNetwork(n_classes=n_classes)
    caps_model.load_state_dict(torch.load("Best_malware_capsule_model_grey.pth"))
    caps_model.eval()
    rf_model = pickle.load(open("RF_malware_feature_np.pkl", "rb"))
    scaler = pickle.load(open("scaler (4).pkl", "rb"))
    return caps_model, rf_model, scaler

# Streamlit app starts here
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model on CPU
#model_path = 'path/to/your/file/Best_malware_capsule_model_grey.pth'
#model = torch.load(model_path, map_location=torch.device('cpu'))

# Load models
caps_model, rf_model, scaler = load_models()
hybrid_model = HybridModel(capsule_model=caps_model, rf_model=rf_model)
caps_model, rf_model, scaler = load_models()
hybrid_model = HybridModel(capsule_model=caps_model, rf_model=rf_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Upload image
image_file = st.file_uploader("Upload an Image (Grayscale, 64x64)", type=["png", "jpg", "jpeg"])
if image_file:
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64)) / 255.0
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Upload CSV
csv_file = st.file_uploader("Upload a CSV File", type=["csv"])
if csv_file:
    test_df = pd.read_csv(csv_file)
    st.write("Uploaded CSV:")
    st.dataframe(test_df.head())

    # Scale the CSV features
    test_df_scaled = scaler.transform(test_df)
    test_df_scaled = pd.DataFrame(test_df_scaled, columns=test_df.columns)

# Predict button
if st.button("Predict"):
    if image_file and csv_file:
        # Predict using both image and CSV
        image_probs = hybrid_model.predict("image", image_tensor)
        csv_probs = hybrid_model.predict("csv", test_df_scaled)
        final_probs = (image_probs + csv_probs) / 2
        final_class = np.argmax(final_probs)
        st.write(f"Predicted Class: {final_class}")
    elif image_file:
        # Predict using image only
        image_probs = hybrid_model.predict("image", image_tensor)
        final_class = np.argmax(image_probs)
        st.write(f"Predicted Class (Image Only): {final_class}")
    elif csv_file:
        # Predict using CSV only
        csv_probs = hybrid_model.predict("csv", test_df_scaled)
        final_class = np.argmax(csv_probs, axis=1)
        st.write("Predicted Classes (CSV Only):")
        st.write(final_class)
    else:
        st.warning("Please upload an image or a CSV file to make predictions.")
