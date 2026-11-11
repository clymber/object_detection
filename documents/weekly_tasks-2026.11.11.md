# Enable Tasks for Next Week

## 1. Check Library Licensing

- Check the license for the Roboflow `supervision` library.
- If its license permits use in the project, proceed with it.
- Otherwise, research alternative open-source object tracking solutions.

## 2. Test Model Deployment on iOS

Test ONNX model deployment using two approaches:

- **Unity + iOS:** Build a small project using Unity and iOS. Pass the models to the MoveAhead team.
- **Native iOS + Core ML:** Bootstrap a native iOS deployment project using Core ML.

### Models to Test

- RF-DETR Small
- RF-DETR Nano
- YOLOX Nano

### Object Tracking

Test the object tracking solution with each of the three models.

## 3. Evaluate On-Device Performance

At minimum, measure the following metrics for each model on both deployment platforms:

- **F1 score**
- **Inference performance (frames per second, FPS)**

| Model | F1 (Unity + iOS) | F1 (Native iOS) | FPS (Unity + iOS) | FPS (Native iOS) |
|---|---|---|---|---|
| RF-DETR Small | | | | |
| YOLOX Nano | | | | |
| RF-DETR Nano | | | | |

## 4. Evaluate Real-Time and Asynchronous Processing

The intended deployment should run **live or near-real-time**.

- Evaluate whether each model can achieve acceptable real-time inference performance on-device.
- If inference is too slow, investigate an **asynchronous processing solution** that allows data to be extracted on-device.

## Expected Deliverables

- Licensing assessment of `supervision` and possible alternatives.
- Unity + iOS prototype and models shared with the MoveAhead team.
- Native iOS + Core ML deployment prototype.
- Object tracking tests for all three models.
- On-device F1 and FPS comparison table.
- Recommendation for real-time or asynchronous on-device processing.