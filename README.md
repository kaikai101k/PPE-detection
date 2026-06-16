# PPE Detection

A real-time PPE compliance monitoring system developed for Georgia Tech's Machine Vision for User Safety VIP project.

The system uses a custom-trained YOLOv8 model deployed on a Raspberry Pi 5 with a Hailo AI HAT+ accelerator to detect personnel and safety glasses in real time. Detection results are displayed through an 8×32 WS2812 LED matrix, providing immediate visual feedback for machine shop safety compliance.

## System Overview

<img width="994" height="464" alt="image" src="https://github.com/user-attachments/assets/80e7504b-e51a-4330-8e86-292b434d58b9" />

## Detection Results

<img width="547" height="466" alt="image" src="https://github.com/user-attachments/assets/1dd79f45-277e-452e-aaba-9445b83fb738" />

## Project Demo

https://youtu.be/vDEzBrBdxic

## LED Status Indicator

### Person + Glasses

<img width="842" height="834" alt="image" src="https://github.com/user-attachments/assets/1639c33d-e065-46b1-8539-96aedbd14487" />

## Technical Challenges

- Collected and labeled a custom PPE dataset
- Trained and evaluated a YOLOv8 object detection model
- Exported the model to ONNX format
- Quantized and compiled the model for Hailo AI acceleration
- Deployed real-time inference on Raspberry Pi 5
- Integrated LED-based safety feedback
  
## Hardware

- Raspberry Pi 5
- Hailo AI HAT+ 2
- Pi Camera Module 3
- 8×32 WS2812 LED Matrix

## Model Performance

| Metric | Value |
|----------|----------|
| Precision | 92.8% |
| Recall | 91.9% |
| mAP50 | 94.0% |
| mAP50-95 | 69.2% |
