# PyTorch-Deep-Learning-Trainer
Abstract Trainer library for Deep Learning Model training - self use

Trainer is an abstract class that has 4 abstract functions

*  fit() - function that trains the model

*  _train() - function for training

*  _val() - function for validation

*  inference() - function for inference using the trained model

This abstract class can also save the parameters in every epoch and automatically select the best performance parameters. It also supports saving the training log at each epoch by simply passing the log string to the function **save_log()**

There is an example Trainer called ClassifierTrainer in the trainer.py
