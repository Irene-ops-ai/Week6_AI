# Edge AI Image Classification & Smart Agriculture IoT System
## Overview
This project demonstrates:
### 1. Edge AI Prototype
A lightweight image classification model trained using TensorFlow, converted to TensorFlow Lite, 
and optimized for deployment on edge devices such as Raspberry Pi or mobile devices.

### 2. AI-Driven IoT Smart Agriculture Simulation
A conceptual design of a smart farming system that uses IoT sensors and machine learning to predict crop yield and automate farm decision-making.

This project showcases how Edge AI and IoT can work together to enable real-time, low-latency, privacy-preserving decision systems.

## Part 1: Edge AI Prototype
### Objective
Build and deploy a lightweight image classifier capable of recognizing objects (e.g., recyclable items), and convert it to TensorFlow Lite for edge deployment.

## Features
- Lightweight CNN model (TensorFlow/Keras)
- Dataset preprocessing + normalization
- Model conversion to TensorFlow Lite (.tflite)
- TFLite inference testing
- Ready for deployment on:
- Raspberry Pi
- Smartphones
- Drones
- Edge IoT devices

## Project Structure
📂 Edge-AI-Project
│
├── model_training.ipynb        # Training notebook
├── recycler_model.tflite       # Exported TensorFlow Lite model
├── sample_inference.py         # TFLite inference script
├── README.md                   # Documentation

## Model Training
### 1. Load and preprocess dataset
A subset of CIFAR-10 was used to simulate a recyclable vs non-recyclable classification task.
Images were normalized and filtered for 2 classes.

### 2. Build a lightweight CNN model
model = models.Sequential([
    tf.keras.layers.Input(shape=(32,32,3)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

## Why modify the model?
To fix the warning/error, we replaced:

❌ input_shape=
✅ tf.keras.layers.Input(shape=...)

## Model Conversion to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("recycler_model.tflite", "wb") as f:
    f.write(tflite_model)
## TFLite Inference Script
interpreter = tf.lite.Interpreter(model_path="recycler_model.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

sample = np.expand_dims(x_test[0], axis=0).astype(np.float32)
interpreter.set_tensor(input_index, sample)
interpreter.invoke()

prediction = interpreter.get_tensor(output_index)
print("Prediction:", prediction)

## Accuracy Metrics
- Training accuracy: ~85–90%
- Test accuracy: varies based on subset used
- Model is optimized for small size + fast inference

## Edge AI Benefits
✅ Real-time predictions
✅ No internet dependency
✅ Improved privacy — data stays on device
✅ Reduced latency
✅ Lower cloud costs
✅ Better reliability for offline environments

- Ideal for:
- Drones
- Smart bins
- Surveillance devices
- Robotics
- IoT systems

## Part 2: AI-Driven IoT Smart Agriculture Simulation
### Objective
Design a conceptual IoT system where sensors collect environmental data, and a 
machine learning model predicts crop yield or recommends farming actions.

## Sensors Used
| Sensor                     | Purpose                      |
| -------------------------- | ---------------------------- |
| Soil Moisture Sensor       | Detects water level          |
| Temperature Sensor (DHT22) | Monitors heat                |
| Humidity Sensor            | Monitors plant transpiration |
| Light Sensor (LDR)         | Sunlight amount              |
| pH Sensor                  | Soil acidity                 |
| Rain Sensor                | Detects rainfall             |
| Camera Module              | Detects crop disease         |

## Proposed AI Model
### Random Forest Regression
✅ Handles nonlinear data
✅ Good with small + noisy datasets
✅ Robust and easy to deploy

### Input Features:
- Soil moisture
- Temperature
- Humidity
- Sunlight
- Rainfall
- pH
- NDVI (from camera)

### Output:
- Estimated crop yield (kg/ha)

## Data Flow Diagram
       ┌───────────────────────┐
       │   IoT Sensors (Farm)  │
       │ Soil, Temp, Light,    │
       │ Humidity, Camera      │
       └───────────┬───────────┘
                   │
                   ▼
       ┌──────────────────────┐
       │   Edge Device (Pi)   │
       │ - Preprocess data    │
       │ - Run ML inference   │
       └───────────┬──────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │   Cloud Dashboard     │
       │ Visualization, Alerts │
       └───────────┬───────────┘
                   │
                   ▼
       ┌──────────────────────┐
       │ Farmer Mobile App     │
       │ Recommendations,      │
       │ Irrigation control    │
       └──────────────────────┘
## Applications in Smart Agriculture
- Automated irrigation
- Disease detection
- Fertilizer recommendations
- Yield prediction
- Real-time farm monitoring
- Water-saving optimization

## Technologies Used
- TensorFlow / Keras
- TensorFlow Lite
- NumPy
- Matplotlib (optional visualization)
- Raspberry Pi (target deployment)
- IoT Sensors (conceptual)
