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
- Last layer HMC
- Zig-zag?
