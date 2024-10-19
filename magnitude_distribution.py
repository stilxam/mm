import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def save_fig(fig, name) -> None:
    """
    Save the figure to the specified path.

    :param name: A string representing the name of the figure.
    :param fig: A matplotlib figure object.
    :return: None
    """
    cwd = Path.cwd()
    path = cwd / "figures" / f"{name}.png"
    if not path.parent.exists(): path.parent.mkdir(parents=True)
    fig.savefig(path)


def analyze_distributions(magnitude_series: pd.Series, significance_level: float = 0.05) -> Tuple[
    List[List[Any]], Dict[str, Dict[str, float]]]:
    """
    Analyze the distribution of the data and test for goodness of fit.

    :param magnitude_series: A pandas Series of earthquake magnitudes.
    :param significance_level: Significance level for the goodness of fit tests.
    :return: A tuple containing the results of the analysis and the output parameters.
    """
    results = []
    output_parameters = {}

    # Uniform distribution
    uniform_fit = stats.uniform(loc=(magnitude_series.min()), scale=magnitude_series.max() - magnitude_series.min())
    uniform_test = stats.kstest(magnitude_series, uniform_fit.cdf)
    uniform_p_value = uniform_test[1]
    uniform_fit_status = "Good fit" if uniform_p_value > significance_level else "Bad fit"
    results.append(["Uniform Distribution", uniform_fit_status, uniform_p_value])
    output_parameters["Uniform Distribution"] = {
        "loc": magnitude_series.min(),
        "scale": magnitude_series.max() - magnitude_series.min()
    }

    # Exponential distribution
    mean_magnitude = np.mean(magnitude_series)
    exponential_fit = stats.expon(scale=1 / mean_magnitude)
    exponential_test = stats.kstest(magnitude_series, exponential_fit.cdf)
    exponential_p_value = exponential_test[1]
    exponential_fit_status = "Good fit" if exponential_p_value > significance_level else "Bad fit"
    results.append(["Exponential Distribution", exponential_fit_status, exponential_p_value])
    output_parameters["Exponential Distribution"] = {
        "estimated_lambda": 1 / mean_magnitude
    }

    # Gamma distribution
    mean_square_magnitude = np.mean([x ** 2 for x in magnitude_series])
    estimated_beta = mean_magnitude / (mean_square_magnitude - mean_magnitude ** 2)
    estimated_alpha = mean_magnitude * estimated_beta
    gamma_fit = stats.gamma(a=estimated_alpha, scale=1 / estimated_beta)
    gamma_test = stats.kstest(magnitude_series, gamma_fit.cdf)
    gamma_p_value = gamma_test[1]
    gamma_fit_status = "Good fit" if gamma_p_value > significance_level else "Bad fit"
    results.append(["Gamma Distribution", gamma_fit_status, gamma_p_value])
    output_parameters["Gamma Distribution"] = {
        "estimated_alpha": estimated_alpha,
        "estimated_beta": estimated_beta
    }

    # Poisson distribution
    poisson_fit = stats.poisson(mu=mean_magnitude)
    poisson_test = stats.kstest(magnitude_series, poisson_fit.cdf)
    poisson_p_value = poisson_test[1]
    poisson_fit_status = "Good fit" if poisson_p_value > significance_level else "Bad fit"
    results.append(["Poisson Distribution", poisson_fit_status, poisson_p_value])
    output_parameters["Poisson Distribution"] = {
        "estimated_lambda": mean_magnitude
    }

    # Normal distribution
    estimated_std_dev = mean_square_magnitude - mean_magnitude ** 2
    normal_fit = stats.norm(loc=mean_magnitude, scale=estimated_std_dev)
    normal_test = stats.kstest(magnitude_series, normal_fit.cdf)
    normal_p_value = normal_test[1]
    normal_fit_status = "Good fit" if normal_p_value > significance_level else "Bad fit"
    results.append(["Normal Distribution", normal_fit_status, np.format_float_scientific(normal_test[1], precision=2)])
    output_parameters["Normal Distribution"] = {
        "estimated_mean": mean_magnitude,
        "estimated_std": estimated_std_dev
    }
    return results, output_parameters


def generate_qq_plot(in_series: pd.Series, output_parameters: Dict[str, Dict[str, float]], case: str) -> None:
    """
    Generate a Q-Q plot of the data against the fitted distribution.
    :param in_series: input time series data
    :param output_parameters: parameters of the fitted distributions
    :param case: A string representing the case of earthquakes to analyze.
    :return:
    """

    uniform_dist = stats.uniform(loc=output_parameters["Uniform Distribution"]["loc"],
                                 scale=output_parameters["Uniform Distribution"]["scale"])
    exp_dist = stats.expon(scale=1 / output_parameters["Exponential Distribution"]["estimated_lambda"])
    gamma_dist = stats.gamma(a=output_parameters["Gamma Distribution"]["estimated_alpha"],
                             scale=1 / output_parameters["Gamma Distribution"]["estimated_beta"])
    poisson_dist = stats.poisson(mu=output_parameters["Poisson Distribution"]["estimated_lambda"])
    normal_dist = stats.norm(loc=output_parameters["Normal Distribution"]["estimated_mean"],
                             scale=output_parameters["Normal Distribution"]["estimated_std"])

    distributions = [exp_dist, gamma_dist, normal_dist, poisson_dist, uniform_dist]
    names = ["Exponential", "Gamma", "Normal", "Poisson", "Uniform"]
    fig, ax = plt.subplots(nrows=len(distributions), ncols=1, figsize=(10, 5 * (len(distributions) + 1)))
    for i, dist in enumerate(distributions):
        stats.probplot(in_series, dist=dist, plot=ax[i])
        ax[i].set_title(f"Q-Q Plot of {names[i]}")
        ax[i].set_xlabel(f"Theoretical Quantiles ({names[i]})")
        ax[i].set_ylabel(f"Sample Quantiles")

    plt.suptitle(f"Q-Q Plot of Eaurthquake {case} vs Theoretical Distributions", fontsize=16)
    save_fig(fig, f"qq_plot_{case}")
    plt.show()


def overlay_distribution_plot(in_series: pd.Series, output_parameters: Dict[str, Dict[str, float]], case: str) -> None:
    """
    Overlay the distribution plots of the fitted distributions and the data.

    :param in_series: A pandas Series of time differences.
    :param output_parameters: A dictionary of output parameters for the fitted distributions.
    :param case: A string representing the case of earthquakes to analyze.
    :return: None
    """
    n = len(in_series)
    exp_dist = np.random.exponential(output_parameters["Exponential Distribution"]["estimated_lambda"], n)
    gamma_dist = np.random.gamma(output_parameters["Gamma Distribution"]["estimated_alpha"],
                                 1 / output_parameters["Gamma Distribution"]["estimated_beta"], n)
    normal_dist = np.random.normal(output_parameters["Normal Distribution"]["estimated_mean"],
                                   output_parameters["Normal Distribution"]["estimated_std"], n)
    poisson_dist = np.random.poisson(output_parameters["Poisson Distribution"]["estimated_lambda"], n)
    uniform_dist = np.random.uniform(output_parameters["Uniform Distribution"]["loc"],
                                     output_parameters["Uniform Distribution"]["scale"], n)

    distributions = [exp_dist, gamma_dist, normal_dist, poisson_dist, uniform_dist]
    names = ["Exponential", "Gamma", "Normal", "Poisson", "Uniform"]

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(10, 25))

    for i, dist in enumerate(distributions):
        sns.histplot(dist, kde=True, ax=ax[i])
        sns.histplot(in_series, kde=True, ax=ax[i])
        ax[i].legend(["Fitted Distribution", "Data"])
        ax[i].set_title(f"{names[i]} Distribution")
        ax[i].set_xlabel("Time Difference")
        ax[i].set_ylabel("Frequency")

    fig.suptitle(
        f"Overlay of the Distribution Earthquake {case} with Theoretical Distributions  ",
        fontsize=16)

    save_fig(fig, f"overlay_dist_{case}")
    plt.show()


def big_magnitude_probability(distribution_type: str, max_magnitude: float,
                              distribution_params: Dict[str, Dict[str, float]]) -> float:
    """
    Calculate the probability of a large earthquake occurring.

    :param distribution_type: Type of the distribution.
    :param max_magnitude: Maximum magnitude of the earthquake.
    :param distribution_params: A dictionary of output parameters for the fitted distributions.
    :return: Probability of a large earthquake occurring.
    """
    if distribution_type == "Uniform Distribution":
        return 1 - stats.uniform.cdf(max_magnitude, loc=distribution_params["Uniform Distribution"]["loc"],
                                     scale=distribution_params["Uniform Distribution"]["scale"])
    elif distribution_type == "Exponential Distribution":
        return 1 - stats.expon.cdf(max_magnitude,
                                   scale=1 / distribution_params["Exponential Distribution"]["estimated_lambda"])
    elif distribution_type == "Gamma Distribution":
        return 1 - stats.gamma.cdf(max_magnitude, a=distribution_params["Gamma Distribution"]["estimated_alpha"],
                                   scale=1 / distribution_params["Gamma Distribution"]["estimated_beta"])
    elif distribution_type == "Normal Distribution":
        return 1 - stats.norm.cdf(max_magnitude, loc=distribution_params["Normal Distribution"]["estimated_mean"],
                                  scale=distribution_params["Normal Distribution"]["estimated_std"])
    else:
        return -1


def magnitude_analysis(earthquake_data: pd.DataFrame, threshold_magnitude: float = 5) -> None:
    """
     Analyze the magnitude data and determine the best fit distribution and the probability of a large earthquake.

     :param earthquake_data: A pandas DataFrame containing earthquake data.
     :param threshold_magnitude: The threshold magnitude to calculate the probability of a large earthquake.
     """
    results, distribution_params = analyze_distributions(earthquake_data["mag"])
    p_values = np.array([result[2] for result in results])

    best_fit_index = np.argmax(p_values)
    best_fit_distribution = results[best_fit_index]
    probability = big_magnitude_probability(best_fit_distribution[0], threshold_magnitude, distribution_params)

    print(f"The best fit distribution for the magnitude of the earthquakes is {best_fit_distribution[0]} with a p-value of {best_fit_distribution[2]}")
    print(f"The estimated parameters are: {distribution_params[best_fit_distribution[0]]}")
    print(f"The probability of a large earthquake occurring is {probability}")

    generate_qq_plot(earthquake_data["mag"], distribution_params, "Magnitude")
    overlay_distribution_plot(earthquake_data["mag"], distribution_params, "Magnitude")


if __name__ == "__main__":
    earthquake_df = pd.read_csv("data/japan_n_skorea.csv")
    magnitude_analysis(earthquake_df)
