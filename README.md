# ICT+ Uncertainty with forams repo

This repo contains scripts for training and analysis using different uncertainty quantification methods on a four-class foraminifera classification problem.

For model training, use the *train_model.py* script: 
- Ensembling:
  - Run ```train_model.py --random_seed 42``` with different seeds. We used seeds 0, 1, ..., 9 in our experiments.
- Monte Carlo Dropout and Test time data augmentation (TTA) does not require special treatment:
  - Run ```train_model.py```
- Stochastic Weight Averaging - Gaussian (SWAG):
  - Use a trained model as a starting poing (```my_model.keras```)
  - Re-init training by running ```train_model.py --path_to_weights "my_model.keras" --swag True```. SWAG statistics will be stored in *swag_diagonal_mean.npy* and *swag_diagonal_squared_sum.npy* files.
  - Sample weights and save each weight configuration as .keras models using ```make_models_swag.py --path_to_weights "my_model.keras" --source "path_to_swag_statistics"```
- Stochastic Gradient Langevin Dynamics (SGLD):
  - Run ```train_model.py --sgld True```

Workflow for producing stats and plots:
- run ```save_predictions_xxx.py``` to produce npy-files with predictions, labels and filenames for the test data.
- run ```eval_predictions.py``` to produce csv-files with summary stats and individual stats for the test results.
- run ```eval_calibration.py```to produce calibration plots and calibration error stats.
- run ```eval_against_experts.py```to produce csv-files with expert comparison stats.
- run ```visualize_uncertainty.py``` to produce uncertainty distrubtion plots++

Other scripts:
- ```compute_correlations.py``` to compute correlations between outputs of two methods.
- ```visualize_augmentations.py``` to visualize the effect of data augmentations.
