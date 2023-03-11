# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change. 

Please note we have a [code of conduct](./CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

## Pull Request Process

1. Ensure tests pass before submitting a pull request.
1. For non-trivial changes, ensure new tests are present before submitting a pull request.
1. Update the README.md with details of changes to the interface, this includes new environment 
   variables, exposed ports, useful file locations and container parameters.
1. You may merge the Pull Request in once you have the sign-off of one other developer, or if you 
   do not have permission to do that, you may request the reviewer to merge it for you.

## Local setup

1. Fork this repository.
2. Clone the forked repository.
3. Change to the cloned directory.
4. Ensure [`poetry`](https://python-poetry.org/docs/#installation) is installed.
5. Run `poetry install`.
6. Run `nox -s test` to run all tests.

## Project layout

The project strucute follows this pattern:

```
   pyproject.toml # The repository toml for setup instructions
   mkdocs.yml     # The configuration file
   docs/
      index.md    # The documentation homepage.
      ...         # Other markdown pages, images and other files
   genai/
      TODO        # Fill out as repo structure solidifies
   tests/      # Unittests for the library
```
