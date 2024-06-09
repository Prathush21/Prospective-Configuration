#Codebase to Reproduce the Experiments

We forked the github code repository ([YuhangSong/Prospective-Configuration](https://github.com/YuhangSong/Prospective-Configuration)) provided in [1] and made the necessary modifications to run our 4 experiments.

Our code submission in Moodle has the following folder structure. Here, we have only included the files that we have either modified or newly created for running our 4 experiments.

For each experiment, we created 2 yaml files—one tailored for the CIFAR10 dataset and the other for the Imagenette dataset. These yaml files encapsulate essential configurations such as model architecture, layer parameters, training methods (BP or PC), and hyperparameters for grid search.  Additionally, we also created a dedicated utils.py file for each experiment, which mainly contains the create_model function that has the code to generate the PyTorch sequential models using the configuration specified in the corresponding yaml file.

For the integration of Imagenette dataset, we created a CustomDataset.py file (inside test_2 folder) and dataset_utils_imagenette.py file that contain contain code crucial for transforming the raw Imagenette dataset into a compatible format for the core functions of our adapted codebase.

To run each experiment we used the run_experiment.ipynb notebook. In this notebook, we first clone our forked and modified repository (<https://github.com/Prathush21/Prospective-Configuration.git>) to access and load all the necessary experiment related yaml and python files. Following this, we set specific configurations, including the target folder for storing experiment results. Subsequently, to visualize and analyze these results, we rely on the plot_results.ipynb notebook.

To replicate any of the four experiments, one need only execute the run_experiment.ipynb and plot_results.ipynb notebooks on Google Colab equipped with a T4 GPU. No additional file downloads are necessary, as all requisite files are accessed via cloning our repository.
