# ICT+ Uncertainty with forams repo

This repo provides training script for xx different uncertainty methods
To train models using different uncertainty methods:
- Ensembling:
  - run ```train_model.py``` with different seeds 0, 1, ..., 9
- Monte Carlo Dropout:
  - run ```train_model.py```
- Stochastic Weight Averaging - Gaussian (SWAG):
  - Complete training using ```train_model.py```to create a starting poing (```my_model.keras```)
  - Re-init training by runnning ```train_model.py --path_to_weights "my_model.keras" --swag True```
  - Sample weights and save as .keras models using ```make_models_swag.py```
- Stochastic Gradient Langevin Dynamics (SGLD)
  - Run ```train_model.py --sgld True```
- Test time data augmentation (TTA)
  - Run ```train_model.py```

Workflow for producing stats and plots:
- run ```save_predictions_xxx.py``` to produce npy-files with predictions, labels and filenames
- run ```eval_predictions.py``` to produce csv-files with summary stats and individual stats
- run ```eval_calibration.py```to produce calibration plots and stats
- run ```eval_against_experts.py```to produce csv-files with expert stats
- run ```visualize_uncertainty.py``` to produce uncertainty plots

Other scripts:
- ```compute_correlations.py``` to compute correlations between predictions.
- ```visualize_augmentations.py``` to visualize data augmentations used.
