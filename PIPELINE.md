# Machine Learning Pipeline

## Components
### [MLFlow](https://www.mlflow.org/docs/latest/index.html)
Stores runs containing information about previously executed ML tasks (e.g. data loading, training, inference),
and stores trained ML models for inferencing.

### [NGAS](https://github.com/ICRAR/ngas)
Provides a storage mechanism for MLFlow to store data and models. Requires Docker.

### [DALIUGE](https://github.com/ICRAR/daliuge)
Executes data loading, training, and inference workflows.

## Tasks
Atomic tasks that accept input from one or more Components, and produce outputs
that are consumed by other Components.

These will be composed together as components in a DALIUGE workflows.
### Load Raw Source Files
Load source files onto the NGAS server.
- Inputs
  - Raw source files
- Outputs
  - Raw source files (NGAS)
  - Raw source files run (MLFlow)

![Load raw source files image](./pipeline_images/load_raw_source_files.drawio.svg)

### Preprocess Raw Source Files
Preprocess a set of loaded source files, from NGAS, and store them back on NGAS.
- Inputs
  - Raw source files (NGAS)
  - Preprocess settings
- Outputs
  - Preprocessed files (NGAS)
  - Preprocess files run (MLFLow)

![Preprocess raw source files image](./pipeline_images/preprocess_raw_source_files.drawio.svg)

### Train Model
Take a set of preprocessed data and produce a model from it.
- Input
  - Preprocessed files (NGAS)
  - Model train parameters
- Output
  - Trained model (MLFlow)
  - Training run (MLFlow)

![Train model image](./pipeline_images/train_model.drawio.svg)

### Retrain Model
Take a model, train it on more data, and produce a new model from it.
- Input
  - Trained model (MLFlow)
  - Preprocessed files (NGAS)
  - Model train parameters
- Output
  - Trained model (MLFlow)
  - Training run (MLFlow)

![Retrain model image](./pipeline_images/retrain_model.drawio.svg)

### Inferencing
Take a trained model and a set of preprocessed data and produce an inference output from it
- Input
  - Preprocessed files (NGAS)
  - Trained model (MLFlow)
- Output
  - Inferencing Output (NGAS)
  - Inferencing run (MLFlow)

![Inferencing image](./pipeline_images/inferencing.drawio.svg)

## Web server
Results generated via inferencing or training should be accessible via a web server.
This server will be publicly accessible and will serve files from the MLFlow runs or NGAS server.

## Notes
- MLFlow can support writing artifacts to different types of backend storage, we can probably set it up to use NGAS as a backend.
- MLFlow stores models with versioning, so we can update an existing model with re-trained versions, and all downstream inferencing tasks will use the updated model automatically.