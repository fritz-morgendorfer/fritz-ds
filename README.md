# Fritz-ds

This project consists of two packages: fritz-ds-lib and fritz-ds-example.

fritz-ds-lib is a library defining useful for data science project classes.</br>
fritz-ds-example contains some example of usage on known ds datasets.

## Installment

To create a conda environment, run

```sh
conda env create
```

To activate it:

```sh
conda activate fritz-ds
```

## Usage

The project's library has been installed together with the environment.
The application can be started with a CLI command 'pipe'.

The example package contains configs for two datasets: 'Titanic' and 'House prices'.

### Train

For example, for the 'Titanic' dataset run

```sh
pipe --cfg src/fritz_ds_example/titanic/app.yaml train
```

To run any command with another dataset just change the name of the folder.


### Predict

```sh
pipe --cfg src/fritz_ds_example/titanic/app.yaml predict
```

### Evaluate

```sh
pipe --cfg src/fritz_ds_example/titanic/app.yaml evaluate
```

### Cross-validation

```sh
pipe --cfg src/fritz_ds_example/titanic/app.yaml cv
```

### The --do-all option

All four commands listed above can be used with the `--do-all` flag.</br>
In this case not only the model chosen in the app config will be used,
but all models found in the `model_cfg_folder` directory.
