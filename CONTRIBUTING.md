# Contributing

In order to contibute to this repository you will need developer access to this repo. To know more about the project go to the [README](README.md) first.


## Install Dev environment

From the root of this repo, run
`conda env create -f environment_dev.yaml`

## Pre-commit hooks

Pre-commits hooks have been configured for this project using the [pre-commit](https://pre-commit.com/) library:

- [black](https://github.com/psf/black) python formatter
- [flake8](https://flake8.pycqa.org/en/latest/) python linter
- [isort](https://pypi.org/project/isort/) sorts imports

To get them going on your side, make sure to have python installed, and run the following
commands from the root directory of this repository:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

# Git conventions

- The section relies on the [Commit Message Guidelines](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit)
- It provides conventions to write commits messages based on the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)

- It aims to :
    - Get a well-structured and easily understandable git history
    - Generate changelogs easily for each release since we can use scripts that parse the commit messages


The commit messages must have the following structure :

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

- `<type>` section :
    - It is mandatory
    - It must be one of the following :
        - build: Changes to our deployment configuration (e.g. docker, requirements)
        - ci : Changes to our CI configuration files and scripts
        - chore: Changes not linked to CI / build or the code (e.g. add issue templates)
        - docs : Documentation changes
        - feat : A new feature
        - fix : A bug fix
        - perf : A code change that improves performance
        - revert: Linked to a revert of a commit
        - refactor : A code change that neither fixes a bug nor adds a feature

        - style : Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
        - test : Adding missing tests or correcting existing tests

- `<scope>` section:
    - optional
    - describes module affected by changes

- `<subject>` section :
    - It is mandatory
    - It contains a succinct description of the change
    - Few recommendations about the subject :
        - use the imperative, present tense: "change" not "changed" nor "changes"
        - don't capitalize the first letter
        - no dot (.) at the end

- <body> section :
    - It is optional
    - It is an extension of the <subject> section used to add a longer description about the changes if relevant

## Coding conventions

Please respect the following conventions to contribute to the code:

- Use hard wrap at 88
- Respect black, isort and flake8 conventions
- Classes' names are Caml case (example: MyClass)
- Functions and variables are in lower case with _ as separator (example: my_function, my_var)
- Names are explicit: avoid mathematical notations, functions' names start with a verb
- Use python typing library: each class and method should be typed (both for inputs and outputs)
- Create custom types if needed
- All classes and functions should have a docstring
- Avoid repeating arguments and returns in docstring (should be explicit with the types) except when it is truly necessary
- A function (or a class) does not take more than 5 arguments, if you need more create a data class
- Avoid dictionaries to pass arguments when possible and prefer dataclasses instead
- Repeat inputs names when calling a function: ex: compute_custom(arg1=arg1, arg2=my_arg2)
- Use list comprehension when it is possible
- Use f strings to add variables in strings: ex: print(f'my var value is {my_var}')
- Use PathLib to handle pathes
- Prefer shutil to os to manage files/ folders creations and deletions
