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
 
galleryset = Path("/home/c743h357/EECS700/Project/GallerySet") #Use a different path or the location of the Gallery Set in your device.
probeset = Path("/home/c743h357/EECS700/Project/ProbeSet") #Use a different path or the location of the Probe Set in your device.

galleryfiles = sorted(galleryset.rglob("*"), key = numerical_sort)
galleryimages = [{"path": g, "data": g.read_bytes()} for g in galleryfiles if g.is_file() and g.suffix.lower() == ".jpg"]
print("Total number of images of Gallery Set =", len(galleryimages))

probefiles = sorted(probeset.rglob("*"), key = numerical_sort)
probeimages = [{"path": p, "data": p.read_bytes()} for p in probefiles if p.is_file() and p.suffix.lower() == ".jpg"]
print("Total number of images of Probe Set =", len(probeimages))

galleryimagedata = [np.array(Image.open(image["path"])) for image in galleryimages]
probeimagedata = [np.array(Image.open(image["path"])) for image in probeimages]

print("Gallery Images' sizes: ", {g.shape for g in galleryimagedata})
print("Probe Images' sizes: ", {p.shape for p in probeimagedata})
