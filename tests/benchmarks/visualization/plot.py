import argparse
from ..core.utils import plot_results, read_results_from_csv

def main():
    parser = argparse.ArgumentParser(
        description="Load results from a CSV and plot them."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file containing the results."
    )
    args = parser.parse_args()

    results = read_results_from_csv(args.csv_file)
    plot_results(results)

if __name__ == "__main__":
    main()
