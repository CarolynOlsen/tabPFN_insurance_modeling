# Insurance Modeling with TabPFN and LightGBM

This repository is used to test TabPFN (a transformer model with in-context learning) for insurance modeling and compare its performance to gradient boosted trees (LightGBM).

## Create and Activate Pipenv Environment

I ran this with Python 13.12, which you'll see in the Pipfile requirements. TabPFN supports other versions, too. If you want to run using the package versions I used, make sure to install Python 13.12 before proceeding.

1. Clone the repository:
    ```sh
    git clone https://github.com/CarolynOlsen/insurance_modeling.git
    cd insurance_modeling
    ```

2. If you have multiple versions of Python installed, you may need to specify the Python version when creating the environment:
    ```sh
    pipenv --python /path/to/your/python3.12
    ```

3. Install the dependencies using `pipenv`:
    ```sh
    pipenv install
    ```

4. Activate the pipenv environment (optional, as `pipenv run` can be used to run commands within the environment):
    ```sh
    pipenv shell
    ```

5. Make the environment accessible to Jupyter notebook as kernel.
    ```sh
    pipenv run python -m ipykernel install --user --name=tabpfn_env --display-name "Python (tabpfn_env)"
    ```

## Usage

### Jupyter Notebook

1. Launch Jupyter Notebook:
    ```sh
    pipenv run jupyter notebook
    ```

2. Open the [modeling.ipynb](http://_vscodecontentref_/1) notebook and run the cells to perform initial exploration and profiling of the insurance modeling dataset.

### Running Scripts

You can also run any Python scripts directly within the pipenv environment:
    ```sh
    pipenv run python your_script.py
    ```

## Project Structure

- [data](http://_vscodecontentref_/2): Directory containing the dataset files.
- [modeling.ipynb](http://_vscodecontentref_/3): Jupyter notebook for initial exploration and profiling of the dataset.
- [Pipfile](http://_vscodecontentref_/4): Pipenv environment configuration file.
- `requirements.txt`: List of required Python packages (for reference).

## License

This project is licensed under the MIT License. See the LICENSE file for details.