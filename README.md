---

# Skiza-App: Speech Emotion Recognition for Kenyan Swahili

Welcome to Skiza-App! This project focuses on developing and deploying a model to recognize emotions from speech, specifically tailored for Kenyan Swahili. The model leverages advanced machine learning techniques and is deployed using Streamlit for the web interface and user interaction.

## **Table of Contents**

1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Model Evaluation](#model-evaluation)
5. [Deployment](#deployment)
6. [Contributing](#contributing)
7. [License](#license)

## **Installation**

To get started with Skiza-App, follow these steps to set up your environment and install the necessary dependencies:

### **1. Clone the Repository**

```bash
git clone https://github.com/estherdatascientist/Swahili-Speech-Emotion-Recognition-System.git
cd Swahili-Speech-Emotion-Recognition-System
```

### **2. Create a Virtual Environment**

It’s recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### **3. Install Dependencies**

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### **4. Download and Prepare Data**

Ensure you have the Swahili speech dataset. Update the `data_dir` path in the configuration files to point to your dataset location.

### **5. Install Streamlit**

If not included in `requirements.txt`, you may need to install Streamlit separately:

```bash
pip install streamlit
```

## **Usage**

### **1. Training the Model**

To train the model, open and run the Jupyter notebook `main.ipynb`. This notebook will load the dataset, preprocess the audio files, extract features, and train various models. The best-performing model (Stacking Model with KNN as the meta-learner) will be saved in the `models` directory.

```bash
jupyter notebook main.ipynb
```

### **2. Running the Streamlit App Locally**

To start the Streamlit app locally, use:

```bash
streamlit run app.py
```

This will open a new browser tab with the Streamlit app, allowing you to upload audio files and get emotion predictions.

### **3. Accessing the Web Version**

You can also access the Skiza-App online at the following URL:

- [Skiza-AI Web App](https://skiza-ai.streamlit.app)

This web version provides the same functionality as the local app, allowing you to upload audio files, folders, or long audio files for emotion analysis.

## **Features**

- **Emotion Recognition**: Detects emotions from audio clips in Kenyan Swahili.
- **Real-time Predictions**: Provides immediate feedback through the Streamlit interface.
- **User-friendly Interface**: Streamlit provides a simple and intuitive web interface for uploading and analyzing audio files.
- **Model Evaluation**: Includes model evaluation metrics such as accuracy, confusion matrix, and ROC curves.

## **Model Evaluation**

The Stacking Model with KNN as the meta-learner achieved an accuracy of 83%, making it the best performer among the tested models. For details on model performance and evaluation, refer to the `evaluation` module in the codebase.

## **Deployment**

Skiza-App is deployed using:

- **Streamlit**: For building the web interface and allowing users to upload and analyze audio files.
- **Streamlit Server**: Manages the web app environment and user interactions.

## **Contributing**

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Please make sure to adhere to the coding standards and include tests for new features.

## **Acknowledgments**

Special thanks to the data collection participants and the following contributors for their valuable input and support:

- [Doreen Wanjiru](https://github.com/DoreenMolly)
- [Gregory Mikuro](https://github.com/gregorymikuro)
- [Samuel Hellen](https://github.com/samuelhellen)
- [Ian Korir](https://github.com/SirIan71)

## **License**

This project is licensed under the GNU General Public License (GPL). See the [LICENSE](LICENSE) file for details.

Thank you for using and contributing to Skiza-App!

---
