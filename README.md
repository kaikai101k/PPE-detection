# PPE Detection

A real-time PPE compliance monitoring system developed for Georgia Tech's Machine Vision for User Safety VIP project.

The system uses a custom-trained YOLOv8 model deployed on a Raspberry Pi 5 with a Hailo AI HAT+ accelerator to detect personnel and safety glasses in real time. Detection results are displayed through an 8×32 WS2812 LED matrix, providing immediate visual feedback for machine shop safety compliance.

## Project Demo

https://youtu.be/vDEzBrBdxic

## System Overview

<img width="983" height="462" alt="image" src="https://github.com/user-attachments/assets/1d042d5a-9927-45fb-bcbe-d56f1cc5e899" />


## Detection Results

<img width="547" height="466" alt="image" src="https://github.com/user-attachments/assets/1dd79f45-277e-452e-aaba-9445b83fb738" />

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
