# Insurance Loss Prediction: TabPFN vs LightGBM

This project explores using TabPFN (Transformer that solves small tabular classification Problems in a Second) for insurance loss prediction and compares its performance to LightGBM.

## Summary
I compared a tuned LightGBM pure premium model to TabPFN models. For the TabPFN modeling I took two approaches:
- Pure premium a.k.a loss cost modeling, where we use a single model to predict policy loss. I wanted to try this because I thought this would be the toughest for TabPFN to compete against LightGBM on, because the target has a compound distribution. It's heavily zero-inflated, with a long right tail. With LightGBM we can handle that explicitly with a Tweedie distribution, but that isn't possible with TabPFN.
- Separate frequency and severity models. Ok, so maybe trying to get TabPFN to understand a Tweedie distribution is a bit mean. What if we have it focus separately on frequency and severity? 

I kept evaluation super light because this is a quick test of this new modeling approach, so I looked only at RMSE and D2 Tweedie Score.

Overall I was impressed with TabPFN's predictions. Both of the approaches with TabPFN were pretty comparable -- very similar RMSE to the LightGBM model, but worse D2 Tweedie score. On RMSE, the pure premium model slightly edged out the separate frequency and severity models. 

Usability findings:
- Dependencies were a headache – I spent more time resolving conflicts than modeling
- Inference is slow and GPU-intensive, and made my 2-year-old gaming machine cry
- Skip the auto ensembling for now; it took forever without an improvement

## What is TabPFN?

TabPFN is a transformer model that uses in-context learning for tabular data prediction. Unlike traditional machine learning models that require iterative training, TabPFN comes pre-trained and learns patterns from data examples in a single forward pass, similar to how large language models process text. Developed by researchers at the University of Freiburg, it offers rapid inference while maintaining competitive performance with state-of-the-art AutoML systems.

## Setup

### Requirements
- Python 3.12 (other versions supported - see TabPFN documentation)
- CUDA-capable GPU recommended

### Environment Setup

This project uses Pipenv for dependency management. 

1. Create and activate a Pipenv environment:
```sh
pipenv install
pipenv shell
```

2. Install `tabpfn-community` package - installation instructions at [TabPFN Community Repository](https://github.com/PriorLabs/tabpfn-community)
   - Keep in mind that when running a pip install in pipenv, you need to start the command `pipenv run pip install...`

3. Set up Jupyter kernel (optional):
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