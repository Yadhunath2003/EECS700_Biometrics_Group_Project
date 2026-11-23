import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
import os
import re
from pathlib import Path
from PIL import Image

def numerical_sort(path):
    number = [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(path))]
    return number
 
# Chetan's path:
# galleryset = Path("/home/c743h357/EECS700/Project/GallerySet") #Use a different path or the location of the Gallery Set in your device.
# probeset = Path("/home/c743h357/EECS700/Project/ProbeSet") #Use a different path or the location of the Probe Set in your device.

# Yadhu's path:
galleryset = (
    Path(r"C:\\Users\\yadhu\\Desktop\\Biometrics_Project\\EECS700_Biometrics_Group_Project\\GallerySet"),
    Path(r"C:\\Users\\yadhu\\Desktop\\Biometrics_Project\\EECS700_Biometrics_Group_Project\\__GallerySet"),
)
probeset = (
    Path(r"C:\\Users\\yadhu\\Desktop\\Biometrics_Project\\EECS700_Biometrics_Group_Project\\ProbeSet"),
    Path(r"C:\\Users\\yadhu\\Desktop\\Biometrics_Project\\EECS700_Biometrics_Group_Project\\__ProbeSet"),
)

galleryfiles = sorted((g for folder in galleryset for g in folder.rglob("*")), key = numerical_sort)
galleryimages = [{"path": g, "data": g.read_bytes()} for g in galleryfiles if g.is_file() and g.suffix.lower() == ".jpg"]
print("Total number of images of Gallery Set =", len(galleryimages))

probefiles = sorted((p for folder in probeset for p in folder.rglob("*")), key = numerical_sort)
probeimages = [{"path": p, "data": p.read_bytes()} for p in probefiles if p.is_file() and p.suffix.lower() == ".jpg"]
print("Total number of images of Probe Set =", len(probeimages))

galleryimagedata = [np.array(Image.open(image["path"])) for image in galleryimages]
probeimagedata = [np.array(Image.open(image["path"])) for image in probeimages]

print("Gallery Images' sizes: ", {g.shape for g in galleryimagedata})
print("Probe Images' sizes: ", {p.shape for p in probeimagedata})
