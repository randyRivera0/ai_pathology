# Implemented architecture

The demo is a layered modular monolith. All layers run in one Python process,
but each layer has a distinct responsibility.

```text
Browser
   |
NiceGUI presentation
   |
InferenceService
   |
ImageClassifier protocol
   |
ResNet50ColonClassifier adapter
   |
TensorFlow/Keras .keras checkpoint
```

## Presentation

`frontend/main.py` owns upload interaction, image preview, transient UI state,
error messages, and result rendering. It calls the inference service instead of
performing image preprocessing or TensorFlow operations itself.

Inference is dispatched with `asyncio.to_thread` so the blocking model call does
not block NiceGUI's event loop. This is local thread orchestration, not a job
queue or distributed worker system.

## Inference service

`backend/inference.py` validates and decodes bounded PNG or JPEG bytes. It calls
an `ImageClassifier` and converts ordered model scores into a framework-neutral
`PredictionResult`.

The service obtains class ordering from model metadata. This prevents the UI or
service from independently reconstructing the training generator's mapping.

## Model contract and adapter

`backend/models/protocol.py` defines the framework-neutral behavior and metadata
required by the application. `ResNet50ColonClassifier` is the default adapter.
It owns TensorFlow loading, explicit Caffe-style preprocessing, SHA-256 artifact
verification, shape validation, and sigmoid-output validation. It exposes the
scores in the fixed order benign colon tissue, then colon adenocarcinoma.

The earlier five-class `LC25000Classifier` remains as historical Experiment 1
support but is detached from the default demo. Training notebooks are not
imported by the application, and model artifacts remain outside ordinary Git
history.

When a second validated model exists, it can implement the same protocol. A
registry should be introduced only when the application genuinely needs to
select between two models. A remote adapter can later preserve this contract if
a model needs an isolated runtime or independent scaling.

## Current operational limits

- UI state is process-global and intended for one local user.
- Predictions are serialized by the model adapter's lock.
- Images are held in memory; there is no object storage or job database.
- The model loads on the first prediction rather than at application startup.
- The default demo supports binary colon classification only; the historical
  five-class adapter is not selectable through the UI.
- There is no queue, microservice, autoscaler, or cloud deployment.

Those are deliberate demo constraints, not claims about a production system.
