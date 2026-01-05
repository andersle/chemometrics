"""Helper script to generate wget commands for downloading files."""

import pathlib
import sys

BASE = "!wget https://raw.githubusercontent.com/andersle/chemometrics/main/exercises/{}/{}"


def generate_wget_commands(file_with_files: pathlib.Path):
    exercise = file_with_files.resolve().parent.name
    with open(file_with_files) as filenames:
        for filei in filenames:
            print(BASE.format(exercise, filei.strip()))


def main(file_with_files: str):
    generate_wget_commands(pathlib.Path(file_with_files))


if __name__ == "__main__":
    main(sys.argv[1])
