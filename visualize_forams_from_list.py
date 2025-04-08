import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.utils import lab_to_long, int_to_lab

path = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/data/Man vs machine_Iver_cropped_with_scale/"
display_filenames = False

# shared mistakes and predictions between ensembles and TTA(s=1)
filenames = [   
    'A,2,4,2_1.png',
    'A,2,4,2_27.png',
    'A,4,3,1_25.png',
    'B,2,4,2_18.png',
    'B,2,4,2_5.png',
    'B,2,4,2_6.png',
    'B,5,4,2_46.png',
    'B,5,4,2_68.png',
    'P,1,3,1_9.png',
    'P,3,3,1_27.png',
    ]

preds = [2, 2, 2, 3, 3, 3, 2, 2, 1, 1]

# mistakes by experts, and predictions by ensembles
#filenames = [   
#    'A,1,3,1_1.png',
#    'A,2,4,2_19.png',
#    'A,2,4,2_27.png',
#    'S,1,3,1_10.png',
#    'S,1,3,1_18.png',
#    'S,1,3,1_44.png',
#    'S,1,3,1_45.png',
#    'S,2,4,2_15.png',
#    'S,2,4,2_29.png',
#    'S,3,3,1_1.png',
#    'S,3,3,1_31.png',
#    ]

#preds = [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0]

# list all files in the subfolders
paths = {}
for folder in os.listdir(path):
    if os.path.isdir(path + folder):
        for file in os.listdir(path + folder):
            if file in filenames:
                paths[file] = os.path.join(path, folder, file)

lab_to_long = {'A': 'B. Agglutinated', 'B': 'B. Calcareous', 'S': 'Sediment', 'P': 'Planktic'}

fig, axs = plt.subplots(3, 4, figsize=(13, 10))
for i, file in enumerate(filenames):
    path = paths[file]
    img = np.array(Image.open(path).resize((224, 224)))
    filename = file
    label = filename[0]
    label = lab_to_long[label]
    pred = lab_to_long[int_to_lab[preds[i]]]
    if i in [1, 3, 10]:
        color = 'red'
    else:
        color = 'black'
    axs[i//4, i%4].imshow(img)
    axs[i//4, i%4].axis('off')
    if display_filenames:
        axs[i//4, i%4].set_title(f'Label: {label}\nPred.: {pred}\nFilename: {filename}', fontsize=15, fontweight='bold')
    else:
        axs[i//4, i%4].set_title(f'Label: {label}\nPred.: {pred}', fontsize=15, fontweight='bold')
    # add a red border to the image if the prediction is wrong
    for spine in axs[i//4, i%4].spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
    
for ax in axs.flatten():
    ax.axis('off')
plt.tight_layout()
plt.savefig('forams_from_list.png')
plt.close()
