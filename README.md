# Insurance Loss Prediction: TabPFN vs LightGBM

This project explores using TabPFN (Transformer that solves small tabular classification Problems in a Second) for insurance loss prediction and compares its performance to LightGBM.

## What is TabPFN?

TabPFN is a transformer model that uses in-context learning for tabular data prediction. Unlike traditional machine learning models that require iterative training, TabPFN comes pre-trained and learns patterns from data examples in a single forward pass, similar to how large language models process text. Developed by researchers at the University of Freiburg, it offers rapid inference while maintaining competitive performance with state-of-the-art AutoML systems.

## Setup

### Requirements
- Python 3.12 (other versions supported - see TabPFN documentation)
- CUDA-capable GPU recommended
- `tabpfn-community` package - installation instructions at [TabPFN Community Repository](https://github.com/PriorLabs/tabpfn-community)

### Environment Setup

This project uses Pipenv for dependency management. After installing TabPFN:

1. Create and activate a Pipenv environment:
```sh
pipenv install
pipenv shell
```

2. Set up Jupyter kernel (optional):
```sh
python -m ipykernel install --user --name=tabpfn_env --display-name "Python (tabpfn_env)"
```

## Implementation

The `modeling.ipynb` notebook demonstrates:
- Data preprocessing for TabPFN
- Model implementation using AutoTabPFN
- Comparison with LightGBM baseline
- Performance evaluation using insurance-specific metrics

## Project Structure
```
├── data/               # Dataset directory
├── modeling.ipynb      # Primary modeling notebook
├── Pipfile            # Environment configuration
```

## License
MIT License

## Additional Resources
For questions about TabPFN implementation or updates, please refer to the [TabPFN Repository](https://github.com/PriorLabs/TabPFN).

Note: This implementation explores TabPFN's application in insurance modeling. Results may vary based on specific use cases and data characteristics.