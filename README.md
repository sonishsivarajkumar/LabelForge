# LabelForge

**Lead Maintainer:** Sonish Sivarajkumar

## Overview

LabelForge is an open-source Python platform for programmatic data labeling and weak supervision. It empowers teams to write simple, reusable labeling functions (LFs) that "vote" on unlabeled data, then combines these noisy signals via a probabilistic label model to produce high-quality training labels. With built-in analytics, a CLI, and an optional web UI, LabelForge streamlines the entire data development workflow—no manual annotation headaches, no vendor lock-in.

## Key Features

### 🔧 Labeling Function Engine
- Define LFs in pure Python with a `@lf` decorator
- Support for heuristics: regex, keyword lists, small ML models
- Automatic LF registration and metadata (name, tags, abstain value)

### 🧠 Probabilistic Label Model
- Implements an EM-based generative model to learn LF accuracies and correlations
- Converts noisy LF votes into soft labels for downstream training
- Exposes `.fit()`, `.predict_proba()`, and evaluation metrics

### 🚀 End-to-End Pipeline
```bash
labelforge run --config config.yaml
```
Orchestrates LF execution → label modeling → discriminative model training → evaluation

### 📊 Analytics & Monitoring
- Per-LF coverage, conflict, and accuracy reports
- Conflict matrix heatmaps to identify redundant or low-value LFs
- Drift detection for LF performance over time

### 🎨 Web UI (Optional)
- React + Tailwind interface for LF development, testing, and visualization
- Dataset preview with label overlays and LF vote highlights
- Interactive dashboard for training job monitoring and metrics

## Quick Start

```bash
pip install labelforge
```

```python
from labelforge import lf, LabelModel
from labelforge.datasets import load_example_data

# Define labeling functions
@lf(name="mention_diabetes", tags={"entity": "disease"})
def lf_mention_diabetes(example):
    return 1 if "diabetes" in example.text.lower() else 0

@lf(name="mention_cancer", tags={"entity": "disease"})
def lf_mention_cancer(example):
    return 1 if "cancer" in example.text.lower() else 0

# Load data and apply LFs
data = load_example_data("medical_texts")
label_model = LabelModel()
label_model.fit(data)

# Get probabilistic labels
soft_labels = label_model.predict_proba(data)
```

## Roadmap

- **v1.0 (MVP)**: Core LF API, label model, CLI, detailed docs, GitHub Pages WebUI
- **v1.1**: Discriminative trainer, evaluation dashboard, minimal UI
- **v2.0**: Plugin system, advanced analytics, LLM-enhanced LF suggestions
- **Future**: Real-time streaming support, end-to-end AutoML workflows, multi-modal labeling

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

Apache-2.0 License - see [LICENSE](LICENSE) for details.
