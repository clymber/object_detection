# ENABLE Meeting Takeaways — 25 November 2026

## Key Findings

- YOLOX Nano with Unity achieves the highest FPS for live or near-live object detection among the evaluated configurations.
- YOLOX Nano is therefore the preferred model for the next stage of development.

## Objectives

- Improve the F1 score of YOLOX Nano while maintaining real-time inference performance.
- Use real-world or representative data for model training and evaluation.
- Focus on the detection of bouncing basketballs.

## Action Items

- [ ] Obtain representative datasets from MoveAhead and/or record additional sample videos.
- [ ] Set up CVAT locally using Docker.
- [ ] Investigate CVAT video annotation and dataset export workflows.
- [ ] Develop a reusable and reproducible dataset annotation pipeline.
- [ ] Create representative, annotated training, validation and independent test datasets.
- [ ] Establish the baseline F1 score of the existing YOLOX Nano model.
- [ ] Fine-tune YOLOX Nano using the newly annotated training data.
- [ ] Re-evaluate the F1 score on the independent test dataset.
- [ ] Verify that improved detection accuracy does not compromise real-time performance in Unity.
