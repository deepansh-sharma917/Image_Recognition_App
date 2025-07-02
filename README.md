# Image Recognition App
A simple yet powerful image classification web app built using **TensorFlow**, **Streamlit**, and the **CIFAR-10 dataset**. This app uses a Convolutional Neural Network (CNN) trained from scratch (no pretrained models) to classify images into 10 common categories such as airplanes, cats, ships, and more.

## 🧠 Recognized Classes

The model can classify images into the following 10 categories:

['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

If the model is not confident in its prediction (confidence < 60%), it will return:

> ❌ **"Unknown Object"**
## 🧰 Tech Stack

- Python 3.11
- TensorFlow 2.x
- Keras (via `tensorflow.keras`)
- OpenCV
- Streamlit



# How will you run this app

# 1. Create a virtual environment
Go to the project directory then type 
python -m venv venv
venv\Scripts\activate     
# 2. Install python dependencies
pip install -r requirements.txt

# 3. Train the python model 
Run in the project directory python train_cifar10_model.py

# 4. Run the app
streamlit run app.py

💡 Features

- 📦 Lightweight, no pretrained models used

- 🤖 CNN built and trained from scratch on CIFAR-10

- 🖼️ Upload custom images for classification

- 🛡️ Fallback logic for low-confidence predictions

# See requirements.txt, but here’s a quick list:

- TensorFlow
- opencv-python
- Pillow
- streamlit







