# Architecture

SupervisedAutoML follows a modular pipeline architecture.

Each stage (ingestion → validation → profiling → task detection → preprocessing → splitting → training → tuning → evaluation → selection → prediction) is isolated into its own package under `src/superml_forge/`.

The Streamlit app (`app.py`) serves as the primary user interface, importing directly from these packages.
