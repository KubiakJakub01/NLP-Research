# NLP-Research
This repository serves as a hub for various Natural Language Processing (NLP) research projects, experiments, and implementations.

## Installation

To install and set up the project, you can use the Python Poetry package manager. Follow the steps below:

1. Make sure you have Python installed on your system. You can download it from the official Python website: [python.org](https://www.python.org/downloads/).

2. Install Poetry by running the following command in your terminal or command prompt:

   ```shell
   curl -sSL https://install.python-poetry.org | python3 -
    ```

3. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/KubiakJakub01/NLP-Research.git
   cd NLP-Research
    ```

4. Install the project dependencies using Poetry:

   ```shell
   poetry install --with dev
    ```

5. To activate the virtual environment, run the following command:

   ```shell
   source $(poetry env info --path)/bin/activate
    ```
    Optionaly you can add the following line to your .bashrc:
    ```shell
    alias activate="source $(poetry env info --path)/bin/activate"
    ```
    Then you can activate the virtual environment by running:
    ```shell
    activate
    ```
