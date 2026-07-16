# Frontend Roadmap

## Current scope

Build and validate the NiceGUI interface for the five-class LC25000 workflow.
Preprocessing and inference are mocked in this phase; backend integration and
real model execution are out of scope.

The interface represents these classes:

- Colon adenocarcinoma
- Benign colon tissue
- Lung adenocarcinoma
- Lung squamous cell carcinoma
- Benign lung tissue

## Version 1: workflow prototype

- [x] Add PNG and JPEG upload with validation.
- [x] Show the selected filename and image preview.
- [x] Represent preprocessing as a mock workflow step.
- [x] Represent inference with a deterministic mock result.
- [x] Show the predicted class and confidence.
- [x] Show model name, version, and expected input size.
- [x] Display the research-only, non-clinical-use notice.
- [ ] Manually test the complete workflow with valid PNG and JPEG images.
- [ ] Verify unsupported, empty, and oversized upload behavior.
- [ ] Verify layout and controls at desktop and narrow viewport sizes.
- [ ] Capture a screenshot for the pull request.

## Integration prerequisites

- [ ] Define the frontend-facing inference interface.
- [ ] Establish one shared source of truth for class labels and output order.
- [ ] Replace mock preprocessing without coupling it to UI components.
- [ ] Replace mock prediction without loading model frameworks in the UI layer.
- [ ] Add automated tests and document their commands.

## Future roadmap

### Version 2: visual explanation

- Add Grad-CAM visualization.
- Explain what the heatmap can and cannot demonstrate.

### Version 3: architecture comparison

- Compare predictions and metadata across supported models.
- Present evaluation metrics with consistent definitions.

### Version 4: explainability

- Add complementary explanation methods.
- Communicate uncertainty and method limitations.

### Version 5: research evaluation

- Add dataset-level error analysis and class-wise metrics.
- Add experiment comparison and reproducibility information.
- Keep all outputs explicitly limited to research and educational use.
