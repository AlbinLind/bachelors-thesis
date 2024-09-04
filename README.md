# Business Analytics Bachelor's Thesis
This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
## Job Shop Scheduling
To find more information about the code please take a look at the [`src/README.md`](https://github.com/AlbinLind/bachelors-thesis/blob/master/src/README.md). 

## Development
### Git Fork Workflow
See instructions here [Forking Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)

### Dependencies
1. Make sure you have [poetry](https://python-poetry.org/docs/#installation) installed. The most simple way is to install with `pip install poetry`, but follow the instructions in the link. You can test if you have poetry installed by running `poetry --version`.
2. Open a terminal in the `./src` directory.
3. Run `poetry install` to download the dependencies.
4. Run `poetry shell` to set your terminal to use the newly created virtual environment.
5. To run a script, simply run `python3 path_to_file_to_run.py` in the terminal in which you started the new shell. Alternatively, if you are running from a notebook, point the kernel to the path of the newly created `.venv` folder in `./src/.venv/Scripts/python`.
