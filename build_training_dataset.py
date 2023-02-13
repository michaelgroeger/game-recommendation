#####################################
#  Building a training dataset      #
#  based on the top n played games  #
#####################################
import argparse

from data_processors.Dataset import build_training_datasets

parser = argparse.ArgumentParser(description="Create training datasets")
parser.add_argument("-n", "--n_highest", type=int, help="Select n most played games")
parser.add_argument("--prompts", action="store_true")
parser.add_argument("--no-prompts", dest="prompts", action="store_false")
args = parser.parse_args()


def main():
    subset_game_information, subset_user_game_matrix = build_training_datasets(
        args.n_highest, add_prompts=args.prompts
    )
    print(
        f"Created new datasets with {len(subset_game_information)} games and {len(subset_user_game_matrix)} users."
    )


if __name__ == "__main__":
    main()
