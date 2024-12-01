# mm: Earthquake Analysis Project
author: "Maxwell Litsios (stilxam)" 

This project performs hypothesis testing on earthquake data to analyze patterns and trends. 
The Stochastic Simulations were mostly written in Jax for the sake of practicing how to write vmaps more efficiently.
I understand and am aware that it is not necessarily the most memory efficient approach, however, it was quite fun to write.
## Project Structure

- `hypothesis_testing.ipynb`: Jupyter Notebook containing the main analysis and hypothesis testing.
- `simulation.ipynb`: Jupyter Notebook containing the simulation of earthquakes data.
- `data/`: Directory containing the earthquake data files.
- `figures/`: Directory containing the figures generated during the analysis.

## Requirements

- Python 3.10
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- jax

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/stilxam/mm.git
    cd mm
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Open the Jupyter Notebook:
    ```sh
    jupyter notebook
    ```

2. Navigate to `hypothesis_testing.ipynb` and run the cells to perform the analysis.

