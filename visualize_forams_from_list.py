import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.utils import lab_to_long, int_to_lab

path = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/data/Man vs machine_Iver_cropped/"

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

# list all files in the subfolders
files = []
for folder in os.listdir(path):
    if os.path.isdir(path + folder):
        for file in os.listdir(path + folder):
            if file in filenames:
                files.append(os.path.join(path, folder, file))

fig, axs = plt.subplots(3, 4, figsize=(13, 10))
for i, file in enumerate(files):
    img = np.array(Image.open(file).resize((224, 224)))
    filename = os.path.basename(file)
    label = filename[0]
    label = lab_to_long[label]
    pred = lab_to_long[int_to_lab[preds[i]]]
    axs[i//4, i%4].imshow(img)
    axs[i//4, i%4].axis('off')
    axs[i//4, i%4].set_title(f'Label: {label}\nPred.: {pred}\nFilename: {filename}', fontsize=12, fontweight='bold')
for ax in axs.flatten():
    ax.axis('off')
plt.tight_layout()
plt.savefig('forams_from_list.png')
plt.close()
