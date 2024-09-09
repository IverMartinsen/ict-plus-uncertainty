# ICT+ Uncertainty with forams repo

This repo provides training script for xx different uncertainty methods
- ensembling:
  - run ```train_model.py``` with different seeds 0, 1, ..., 9
- Monte Carlo Dropout:
  - run ```train_model.py```
- SWAG:
  - complete training using ```train_model.py```to create a starting poing (```my_trained_model.keras```)
  - run ```train_with_swag.py``` with ```my_trained_model.keras``` as starting point
- SGLD
- Test time data augmentation

Workflow for producing results and analysis:
- run ```save_predictions_xxx.py``` to produce npy-files with predictions, labels and filenames
- run ```eval_predictions.py``` to produce csv-files with summary stats and individual stats
- run ```eval_calibration.py```to produce calibration plots and stats
- run ```eval_against_experts.py```to produce csv-files with expert stats
- run ```visualize_uncertainty.py``` to produce uncertainty plots
