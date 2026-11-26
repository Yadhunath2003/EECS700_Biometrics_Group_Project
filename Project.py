import numpy as np #This line uses and imports the modules of Numpy for performing and calculating mathematical operations.
import pandas as pd #This line uses and imports the modules of Pandas for collecting and organizing required data.
import matplotlib.pyplot as plt #This line uses and imports the modules of Matplotlib for creating and displaying graphs.
import re #This line uses and imports the modules of RE for searching and matching regular expressions.
from pathlib import Path #This line uses and imports the modules of Pathlib for using and managing paths of huge datasets.
from PIL import Image #This line uses and imports the modules of PIL for opening and manipulating images from huge datasets.
from insightface.app import FaceAnalysis #This line uses and imports the modules of InsightFace for downloading and accessing the pre-trained Deep Learning model like ArcFace ResNet50.
import warnings #This line uses and imports the modules of Warnings for ignoring and suppressing the model's warnings during model execution.
import time #This line uses and imports the modules of Time for finding and measuring the execution time and intervals of the code lines.

"""
These 3 lines that are below this multi-line comment define the numerical_sort() function that numerically sorts the images/files of the Gallery Set and the Probe Set folders.
"""
def numerical_sort(path):
    number = [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(path))] #This line uses the re.split() function to handle regular expressions, split the numbers in the images' names, and extract the numbers as integers.
    return number #This line returns the number.

"""
These 21 lines that are below this multi-line comment define the extraction_embeddings() function that uses the images of the Gallery Set and the Probe Set folders and the pre-trained deep learning model like ArcFace ResNet50 to create embedding vectors and checks if the images are extracted successfully.
"""
def extraction_embeddings(image_list, mod):
    embeddings = [] #This line creates an empty list for the embedding vectors.
    successfulimages = [] #This line creates an empty list for the image paths if the images' facial features are extracted successfully.
    failedimages = [] #This line creates an empty list for the image paths if the images' facial features are not detected properly.
 
    #The for-loop with 14 lines extracts the images' features if the faces are detected. 
    for i in image_list:
        imagepath = i["path"] #This line uses the image path.
        img = Image.open(imagepath).convert("RGB") #This line open the image and converts it into RGB image. 
        img = np.array(img) #This line converts the image into a Numpy array.
        img = img[:, :, ::-1].copy() #This line reverse the image format and creates a copy of the image array.
        faces = mod.get(img) #This line uses the pre-trained deep learning model to detect faces.
	#This if-statement with 3 lines checks if the image has a face.
        if (len(faces) == 0):
            failedimages.append(imagepath) #This line appends or includes the failed images' paths in the failedimages variable.
            continue #This line continues by skipping this condition and checks the next image.
        face = max(faces, key = lambda f: ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))) #This line selects the largest detected face in the box area.
        embedding = face.normed_embedding #This line extracts the normalized embedding vectors of the face.
        embeddings.append(embedding) #This line appends or includes the embedding vectors in the embeddings variable.
        successfulimages.append(imagepath) #This line appends or includes the successful images' paths in the successfulimages variable.
    return (np.array(embeddings), successfulimages, failedimages) #This line returns the embedding vectors in Numpy arrays and 2 lists of image paths.

"""
These 11 lines that are below this multi-line comment define the extraction_ids() function that finds the subject and image IDs.
"""
def extraction_ids(image_paths):
    subjectids = [] #This line creates an empty list for the subject IDs.
    imageids = [] #This line creates an empty list for the image IDs of the respective subjects.
    
    #The for-loop with 5 lines extracts the subject and image IDs from the images' names successfully. 
    for path in image_paths:
        filename = Path(path).stem #This line uses the paths of the images.
        subjectid, imageid = filename.split("_") #This line uses the images' names and the underscore separator to split the subject IDs and image IDs.
        subjectids.append(subjectid) #This line appends or includes the subject IDs in the subjectids variable.
        imageids.append(imageid) #This line appends or includes the image IDs in the imageids variable
    return subjectids, imageids #This line returns the 2 lists of subject IDs and image IDs.
  
"""
These 10 lines that are below this multi-line comment define the binary_csv() function that stores original data of embedding vectors in a DataFrame and binary data of embedding vectors in another DataFrame and uses a separate binarization method.
"""  
def binary_csv(embeddings, subjectids, imageids, name):
    df = pd.DataFrame(embeddings, columns = [(i+1) for i in range(embeddings.shape[1])]) #This line creates a DataFrame to include the embedding vectors and column labels.
    df.insert(0, "ImageID", imageids) #This line inserts a column to store the image IDs.
    df.insert(0, "SubjectID", subjectids) #This line inserts another column to store the subject IDs.
    df.to_csv(f"{name}_embedding.csv", index = False) #This line writes the original DataFrame that is saved in the CSV file and removes the row labels.
    binary = (embeddings >= 0).astype(np.uint8) #This line uses the binarization method like zero-thresholding technique to convert the embedding values into binary values/bits.
    binarydf = pd.DataFrame(binary, columns = [(i+1) for i in range(embeddings.shape[1])]) #This line creates another DataFrame to include the binary templates and column labels.
    binarydf.insert(0, "ImageID", imageids) #This line inserts a column to store the image IDs.
    binarydf.insert(0, "SubjectID", subjectids) #This line inserts a column to store the subject IDs.
    binarydf.to_csv(f"{name}_binary.csv", index = False) #This line writes the binary DataFrame that is saved in the CSV file and removes the row labels.

"""
These 32 lines that are below this multi-line comment define the probe_csv_results() function that uses the probe set's condition to find the score matrix, display the histogram, and calculate and return the decidability index value.
""" 
def probe_csv_results(probeset, imageid):
    scorematrixprobe = np.zeros((probeset.shape[0], probeset.shape[0])) #This line creates a 2D matrix of the given shape with zeros.
    #The nested for-loop with 6 lines iterates to generate gallery and probe images' respective scores that are stored in the score matrix.
    for i in range(probeset.shape[0]):
        probebinary = probeset.iloc[i, 2:] #This line uses the binary values of the probe set's condition.
        for j in range(probeset.shape[0]):
            gal = gallerybinaryreaddf.iloc[j, 2:] #This line uses the binary values of the gallery set's condition.
            hd = np.sum(gal.to_numpy() ^ probebinary.to_numpy()) #This line uses the Hamming Distance that has the XOR operator and finds the score.
            scorematrixprobe[i, j] = hd #This line stores the score in the score matrix.
        
    probedf = pd.DataFrame(scorematrixprobe, index = [f"{sid}_{iid}" for sid, iid in zip(probeset["SubjectID"], probeset["ImageID"])], columns = [f"{sid}_{iid}" for sid, iid in zip(gallerysubjectids, galleryimageids)]) #This line creates a DataFrame to include the scores and row and column labels.
    print(f"Original size of the Score Matrix Img{imageid}:", probedf.shape) #This line prints the original size of the score matrix.
    print(f"Total number of scores of the Score Matrix Img{imageid}:", probedf.size) #This line prints the total number of scores of the score matrix.
    probedf.to_csv(f"ScoreMatrixImg{imageid}.csv", index = True) #This line writes this score matrix that is saved in the CSV file and adds the row labels.
    
    genuinescores = np.diag(scorematrixprobe) #This line finds the genuine scores or the diagonal values of the score matrix.
    nondiagonal = ~np.eye(scorematrixprobe.shape[0], dtype = bool) #This line extracts the non-diagonal values of the score matrix and ignores the diagonal.
    impostorscores = scorematrixprobe[nondiagonal] #This line finds the imposter scores by returning an array of impostor scores of the score matrix.
    decidabilityindex = ((np.sqrt(2) * np.abs(np.mean(genuinescores) - np.mean(impostorscores)))/(np.sqrt((np.std(genuinescores) ** 2) + (np.std(impostorscores) ** 2)))) #This line uses the decidability index value formula to calculate the decidability index value.
    print(f"The d' value of the Score Matrix Img{imageid} =", decidabilityindex, "\n") #This line prints the decidability index value and a new line.
    
    plt.figure(figsize = (8,5)) #This line increases the width and the height of the figure/image.
    plt.hist(genuinescores, bins = 50, density = True, alpha = 0.6, color = "blue", label = "Genuine scores") #This line plots the histogram of the genuine scores.
    plt.hist(impostorscores, bins = 50, density = True, alpha = 0.6, color = "red", label = "Impostor scores") #This line plots the histogram of the impostor scores.
    plt.xlabel("Scores") #This line provides the label of x-axis.
    plt.ylabel("Density") #This line provides the label of y-axis.
    plt.title(f"Score Distributions of the Score Matrix Img{imageid}") #This line provides the title of this figure for the score distributions.
    plt.legend() #This line includes a legend that identifies these score distributions.
    plt.grid(alpha = 0.3) #This line adds visible grid lines to this figure with 30% transparency/visibility.
    plt.savefig(f"Histogram{imageid}.png") #This line saves this figure as the PNG file.
    plt.show() #This line displays this figure.
    return decidabilityindex #This line returns the decidability index value.

galleryset = Path("/home/c743h357/EECS700/Project/GallerySet") #Use a different path or the location of the Gallery Set in your device.
probeset = Path("/home/c743h357/EECS700/Project/ProbeSet") #Use a different path or the location of the Probe Set in your device.

galleryfiles = sorted(galleryset.rglob("*"), key = numerical_sort) #This line recursively retrieves the Gallery Set's images/files and numerically sorts them by the numerical_sort() function.
galleryimages = [{"path": g, "data": g.read_bytes()} for g in galleryfiles if g.is_file() and g.suffix.lower() == ".jpg"][:4] #This line finds the paths and data of the images of Gallery Set.
print("Total number of images of Gallery Set =", len(galleryimages)) #This line prints the total number of images of Gallery Set.

probefiles = sorted(probeset.rglob("*"), key = numerical_sort) #This line recursively retrieves the Probe Set's images/files and numerically sorts them by the numerical_sort() function.
probeimages = [{"path": p, "data": p.read_bytes()} for p in probefiles if p.is_file() and p.suffix.lower() == ".jpg"][:16] #This line finds the paths and data of the images of Probe Set.
print("Total number of images of Probe Set =", len(probeimages)) #This line prints the total number of images of Probe Set.

galleryimagedata = [np.array(Image.open(image["path"])) for image in galleryimages] #This line opens the images of Gallery Set, converts these images into Numpy arrays, and stores their data.
probeimagedata = [np.array(Image.open(image["path"])) for image in probeimages] #This line opens the images of Probe Set, converts these images into Numpy arrays, and stores their data.

print("Gallery Set's Images' size:", {g.shape for g in galleryimagedata}) #This line prints the size of the images of Gallery Set. 
print("Probe Set's Images' size:", {p.shape for p in probeimagedata}, "\n") #This line prints the size of the images of Probe Set and a new line.

model = FaceAnalysis(name = "buffalo_l", providers = ["CPUExecutionProvider"]) #This line uses and loads the pre-trained deep learning model package.
model.prepare(ctx_id = -1) #This line prepares and runs the model on CPU.
print("Pre-trained model ArcFace ResNet50 is loaded successfully.") #This line prints the message if the model is downloaded and used successfully.
warnings.filterwarnings("ignore", category = FutureWarning, module = "insightface.utils.face_align") #This line ignores and suppresses the warnings of the model.

starttime = time.time() #This line records the start time of the embedding of the images of Gallery and Probe Sets. 
galleryembeddings, gallerysuccess, galleryfailure = (extraction_embeddings(galleryimages, model)) #This line uses the extraction_embeddings() function to extract the facial features of the images of Gallery Set.
probeembeddings, probesuccess, probefailure = (extraction_embeddings(probeimages, model)) #This line uses the extraction_embeddings() function to extract the facial features of the images of Probe Set.
endtime = time.time() #This line records the end time of the embedding of the images of Gallery and Probe Sets.

print("Gallery Set's Images' embeddings' size:", galleryembeddings.shape) #This line prints the embeddings' size of the images of Gallery Set.
print("Gallery Set's Images' extraction success paths' count and percentage =", len(gallerysuccess), "or " + str((len(gallerysuccess)/(len(gallerysuccess) + len(galleryfailure))) * 100) + "%") #This line prints the extraction success paths' count and percentage of the images of Gallery Set.
print("Gallery Set's Images' extraction failure paths' count and percentage =", len(galleryfailure), "or " + str((len(galleryfailure)/(len(gallerysuccess) + len(galleryfailure))) * 100) + "%") #This line prints the extraction failure paths' count and percentage of the images of Gallery Set.
print("Probe Set's Images' embeddings' size:", probeembeddings.shape) #This line prints the embeddings' size of the images of Probe Set.
print("Probe Set's Images' extraction success paths' count and percentage =", len(probesuccess), "or " + str((len(probesuccess)/(len(probesuccess) + len(probefailure))) * 100) + "%") #This line prints the extraction success paths' count and percentage of the images of Probe Set.
print("Probe Set's Images' extraction failure paths' count and percentage =", len(probefailure), "or " + str((len(probefailure)/(len(probesuccess) + len(probefailure))) * 100) + "%") #This line prints the extraction failure paths' count and percentage of the images of Probe Set.
print("Total Image Embedding runtime =", (endtime - starttime), "sec or", ((endtime - starttime)/60), "min\n") #This line prints the total execution time of the Image embedding vectors and a new line.

gallerysubjectids, galleryimageids = extraction_ids(gallerysuccess) #This line uses the extraction_ids() function to extract the subject and image IDs of the images of Gallery Set.
probesubjectids, probeimageids = extraction_ids(probesuccess) #This line uses the extraction_ids() function to extract the subject and image IDs of the images of Probe Set.

binary_csv(galleryembeddings, gallerysubjectids, galleryimageids, "GallerySet") #This line uses the binary_csv() function to create the binary templates of Gallery Set.
binary_csv(probeembeddings, probesubjectids, probeimageids, "ProbeSet") #This line uses the binary_csv() function to create the binary templates of Probe Set.

galleryreaddf = pd.read_csv("GallerySet_embedding.csv") #This line reads and stores the GallerySet_embedding.csv file in a separate DataFrame.
probereaddf = pd.read_csv("ProbeSet_embedding.csv") #This line reads and stores the ProbeSet_embedding.csv file in a separate DataFrame.
gallerybinaryreaddf = pd.read_csv("GallerySet_binary.csv") #This line reads and stores the GallerySet_binary.csv file in a separate DataFrame.
probebinaryreaddf = pd.read_csv("ProbeSet_binary.csv") #This line reads and stores the ProbeSet_binary.csv file in a separate DataFrame.

print("Gallery Set's Embeddings' template size:", galleryreaddf.shape) #This line prints the template size of the original embeddings of Gallery Set.
print("Probe Set's Embeddings' template size:", probereaddf.shape) #This line prints the template size of the original embeddings of Probe Set.
print("Gallery Set's Binary Embeddings' template size:", gallerybinaryreaddf.shape) #This line prints the template size of the binary embeddings of Gallery Set.
print("Probe Set's Binary Embeddings' template size:", probebinaryreaddf.shape) #This line prints the template size of the binary embeddings of Probe Set.

scorematrixA = np.zeros((probebinaryreaddf.shape[0], gallerybinaryreaddf.shape[0])) #This line creates the score matrix A or a 2D matrix of the given shape with zeros.

starttime2 = time.time() #This line records the start time of the calculation of the score matrix.
#The nested for-loop with 6 lines iterates to generate gallery and probe images' respective scores that are stored in the score matrix.
for i in range(probebinaryreaddf.shape[0]):
    pro = probebinaryreaddf.iloc[i, 2:] #This line uses the binary values of the probe set.
    for j in range(gallerybinaryreaddf.shape[0]):
        gal = gallerybinaryreaddf.iloc[j, 2:] #This line uses the binary values of the gallery set.
        hd = np.sum(gal.to_numpy() ^ pro.to_numpy()) #This line uses the Hamming Distance that has the XOR operator and finds the score.
        scorematrixA[i, j] = hd #This line stores the score in the score matrix.

Adf = pd.DataFrame(scorematrixA, index = [f"{sid}_{iid}" for sid, iid in zip(probesubjectids, probeimageids)], columns = [f"{sid}_{iid}" for sid, iid in zip(gallerysubjectids, galleryimageids)]) #This line creates a DataFrame to include the score matrix and row and column labels.
print("Original size of the Score Matrix A:", Adf.shape) #This line prints the original size of the score matrix A.
print("Total number of scores of the Score Matrix A:", Adf.size) #This line prints the total number of scores of the score matrix A.
Adf.to_csv("ScoreMatrixA.csv", index = True) #This line writes this score matrix that is saved in the ScoreMatrixA.csv file and adds the row labels.
endtime2 = time.time() #This line records the start time of the calculation of the score matrix.
print("Total Hamming Distance runtime =", (endtime2 - starttime2), "sec or", ((endtime2 - starttime2)/60), "min\n") #This line prints the total execution time of the Score Matrix A and a new line.

genuinescoresA = [] #This line creates an empty list for the genuine scores of the score matrix A.
impostorscoresA = [] #This line creates an empty list for the impostor scores of the score matrix A.

starttime3 = time.time() #This line records the start time of the calculation of the genuine and impostor scores of the score matrix A.
#The nested for-loop with 9 lines enumerates or iterates to store these genuine and impostor scores.
for i, probe_image in enumerate(Adf.index):
    probe_subject = probe_image.split("_")[0] #This line splits the strings/names of the probe images at the underscore and finds the subject IDs of the probe images.
    for j, gallery_image in enumerate(Adf.columns):
        gallery_subject = gallery_image.split("_")[0] #This line splits the strings/names of the gallery images at the underscore and finds the subject IDs of the gallery images.
        #This if-statement with 4 lines checks if the probe images and the gallery images have the same subject IDs.
        if probe_subject == gallery_subject:
            genuinescoresA.append(Adf.iloc[i, j]) #This line appends or includes the genuine scores in the genuinescoresA variable.
        else:
            impostorscoresA.append(Adf.iloc[i, j]) #This line appends or includes the impostor scores in the impostorscoresA variable.

genuinescoresA = np.array(genuinescoresA) #This line converts the list of genuine scores into a Numpy array.
impostorscoresA = np.array(impostorscoresA) #This line converts the list of impostor scores into a Numpy array.
print("Number of genuine scores of the Score Matrix A =", len(genuinescoresA)) #This line prints the number of genuine scores of the score matrix A.
print("Number of impostor scores of the Score Matrix A =", len(impostorscoresA)) #This line prints the number of impostor scores of the score matrix A.

plt.figure(figsize = (8,5)) #This line increases the width and the height of the figure/image.
plt.hist(genuinescoresA, bins = 50, density = True, alpha = 0.6, color = "blue", label = "Genuine scores") #This line plots the histogram of the genuine scores of the score matrix A.
plt.hist(impostorscoresA, bins = 50, density = True, alpha = 0.6, color = "red", label = "Impostor scores") #This line plots the histogram of the impostor scores of the score matrix A.
plt.xlabel("Scores") #This line provides the label of x-axis.
plt.ylabel("Density") #This line provides the label of y-axis.
plt.title("Score Distributions of the Score Matrix A") #This line provides the title of this figure for the score distributions of the score matrix A.
plt.legend() #This line includes a legend that identifies these score distributions.
plt.grid(alpha = 0.3) #This line adds visible grid lines to this figure with 30% transparency/visibility.
plt.savefig("Histogram.png") #This line saves this figure as Histogram.png file.
plt.show() #This line displays this figure.

print("Mean of genuine scores of the Score Matrix A =", np.mean(genuinescoresA)) #This line prints the mean of genuine scores of the score matrix A.
print("Mean of impostor scores of the Score Matrix A =", np.mean(impostorscoresA)) #This line prints the mean of impostor scores of the score matrix A.
print("Standard deviation of genuine scores of the Score Matrix A =", np.std(genuinescoresA)) #This line prints the standard deviation of genuine scores of the score matrix A.
print("Standard deviation of impostor scores of the Score Matrix A =", np.std(impostorscoresA)) #This line prints the standard deviation of impostor scores of the score matrix A.
decidabilityindexA = ((np.sqrt(2) * np.abs(np.mean(genuinescoresA) - np.mean(impostorscoresA)))/(np.sqrt((np.std(genuinescoresA) ** 2) + (np.std(impostorscoresA) ** 2)))) #This line uses the decidability index value formula to calculate the decidability index value of the score matrix A.
print("The d' value of the Score Matrix A =", decidabilityindexA) #This line prints the decidability index value of the score matrix A.
endtime3 = time.time() #This line records the end time of the calculation of the genuine and impostor scores of the score matrix A.
print("Total System Performance runtime =", (endtime3 - starttime3), "sec or", ((endtime3 - starttime3)/60), "min\n") #This line prints the total execution time of the genuine and impostor scores and a new line.

starttime4 = time.time() #This line records the start time of the CMC curve of the score matrix A.
matrixA = np.array(Adf) #This line converts this score matrix into a Numpy array.
rankedvaluesA = np.take_along_axis(matrixA, np.argsort(matrixA, axis = 1), axis = 1) #This line finds and sorts the column indices of rows by these scores in an ascending order.
ratesA = [] #This line creates an empty list for the identification rates.

#The for-loop with 4 lines uses these scores to use the ranks and calculate their corresponding Rank-t Identification Rates of the right matches for the probes.
for t in range(1, matrixA.shape[1] + 1):
    correctmatch = [genuinescoresA[i] in rankedvaluesA[i, :t] for i in range(matrixA.shape[0])] #This line determines the total number of right matches in t identities.
    identificationrate = np.mean(correctmatch) * 100 #This line calculates the Rank-t Identification Rates of the ranks by the proportions of these right matches and the percentages of these proportions.
    ratesA.append(identificationrate) #This line appends or includes these identification rates of these ranks in the ratesA variable.
ratesA = np.array(ratesA) #This line converts the list of identification rates into a Numpy array.
ranksA = np.arange(1, matrixA.shape[1] + 1) #This line creates the Numpy array that contains the ranks.

print(f"Lowest rank of the Score Matrix A: Rank-{np.min(ranksA)}. Its Identification Rate = " + str(ratesA[0]) + "%") #This line prints the lowest rank of the score matrix A and its identification rate.
print(f"Highest rank of the Score Matrix A: Rank-{np.max(ranksA)}. Its Identification Rate = " + str(ratesA[-1]) + "%") #This line prints the highest rank of the score matrix A and its identification rate.

plt.figure(figsize = (8,5)) #This line increases the width and the height of the figure/image.
plt.plot(range(1, matrixA.shape[1] + 1), ratesA, color = "blue") #This line plots the CMC curve by using the Ranks and their Rank-t Identification Rates of the score matrix A.
plt.xlim(1, 1100) #This line sets the range of x-axis from 1 to 1100.
plt.xlabel("Rank (t)") #This line provides the label of x-axis.
plt.ylabel("Rank-t Identification Rate (%)") #This line provides the label of y-axis.
plt.title("Cumulative Match Characteristic (CMC) curve of the Score Matrix A") #This line provides the title of this figure for the Cumulative Match Characteristic (CMC) curve of the score matrix A.
plt.grid() #This line adds visible grid lines to this figure.
plt.savefig("CumulativeMatchCharacteristicCurve.png") #This line saves this figure as CumulativeMatchCharacteristicCurve.png file.
plt.show() #This line displays this figure.
endtime4 = time.time() #This line records the end time of the CMC curve of the score matrix A.
print("Total CMC Curve runtime =", (endtime4 - starttime4), "sec or", ((endtime4 - starttime4)/60), "min\n") #This line prints the total execution time of the CMC curve and a new line.

starttime5 = time.time() #This line records the start time of the ROC curve of the score matrix A.
concatscoresA = np.concatenate((genuinescoresA, impostorscoresA)) #This line concatenates the genuine and impostor scores of the score matrix A.
thresholdsA = np.sort(np.unique(concatscoresA)) #This line removes any duplicate scores and sorts these scores as the thresholds in an ascending order.
FARA = [] #This line creates an empty list for the False Accept Rates.
FRRA = [] #This line creates an empty list for the False Reject Rates.

#The for-loop with 5 lines uses these thresholds to determine the False Accept Rates and the False Reject Rates.
for t in thresholdsA:
    FAR = np.divide(np.sum(impostorscoresA <= t), len(impostorscoresA)) #This line uses the formula of False Accept Rates to calculate the proportions of the falsely accepted impostor scores.
    FRR = np.divide(np.sum(genuinescoresA > t), len(genuinescoresA)) #This line uses the formula of False Reject Rates to calculate the proportions of the falsely rejected genuine scores.
    FARA.append(FAR) #This line appends or includes the False Accept Rates of the score matrix A in the FARA variable.
    FRRA.append(FRR) #This line appends or includes the False Reject Rates of the score matrix A in the FRRA variable.
FARA = np.array(FARA) #This line converts the list of False Accept Rates into a Numpy array.
FRRA = np.array(FRRA) #This line converts the list of False Reject Rates into a Numpy array.
EERindex = np.argmin(np.abs(FARA - FRRA)) #This line finds the absolute differences of False Reject Rates and False Accept Rates and the index of the lowest difference.
EER = np.mean([FARA[EERindex], FRRA[EERindex]]) #This line finds the average of the minimum index's False Reject Rate and False Accept Rate of the score matrix A.
print("False Accept Rate (FAR) of the Score Matrix A =", round(FARA[EERindex], 2)) #This line prints the minimum index's False Accept Rate of the score matrix A.
print("False Reject Rate (FRR) of the Score Matrix A =", round(FRRA[EERindex], 2)) #This line prints the minimum index's False Reject Rate of the score matrix A.
print("Equal Error Rate (EER) of the Score Matrix A =", round(EER, 2)) #This line prints the Equal Error Rate of the score matrix A.
print("Threshold =", thresholdsA[EERindex]) #This line prints the threshold of the Equal Error Rate of the score matrix A.

plt.figure(figsize = (8,5)) #This line increases the width and the height of the figure/image.
plt.plot(FARA, FRRA, label = "ROC Curve")
plt.xlabel("False Accept Rate (FAR)") #This line provides the label of x-axis.
plt.ylabel("False Reject Rate (FRR)") #This line provides the label of y-axis.
plt.title("Receiver Operating Characteristic (ROC) curve of the Score Matrix A") #This line provides the title of this figure for the Receiver Operating Characteristic (ROC) curve of the score matrix A.
plt.scatter(FARA[EERindex], FRRA[EERindex], color = "black", label = "EER (FAR = FRR)", s = 15, zorder = 10) #This line adds a black scatter point at the Equal Error Rate of the ROC curve of the score matrix A.
plt.annotate(f"({round(FARA[EERindex], 2)}, {round(FRRA[EERindex], 2)})", xy = (FARA[EERindex], FRRA[EERindex]), xytext = (5, 5), textcoords = "offset points", fontsize = 8, color = "black", horizontalalignment = "left", verticalalignment = "center") #This line displays the coordinates of the black scatter point of the Equal Error Rate of the ROC curve of the score matrix A.
plt.legend() #This line includes a legend that identifies the ROC curve and the Equal Error Rate of the score matrix A.
plt.grid() #This line adds visible grid lines to this figure.
plt.savefig("ReceiverOperatingCharacteristicCurve.png") #This line saves this figure as ReceiverOperatingCharacteristicCurve.png file.
plt.show() #This line displays this figure.
endtime5 = time.time() #This line records the end time of the ROC curve of the score matrix A.
print("Total ROC Curve runtime =", (endtime5 - starttime5), "sec or", ((endtime5 - starttime5)/60), "min\n") #This line prints the total execution time of the ROC curve and a new line.

starttime6 = time.time() #This line records the start time of the Probe Conditions.
noisyprobe = probebinaryreaddf[probebinaryreaddf["ImageID"] == "img2"] #This line creates a new DataFrame for img2 files or noisy probe images.
profileprobe = probebinaryreaddf[probebinaryreaddf["ImageID"] == "img3"] #This line creates a new DataFrame for img3 files or profile probe images.
blurryprobe = probebinaryreaddf[probebinaryreaddf["ImageID"] == "img4"] #This line creates a new DataFrame for img4 files or blurry probe images.
occludedprobe = probebinaryreaddf[probebinaryreaddf["ImageID"] == "img5"] #This line creates a new DataFrame for img5 files or occluded probe images.

print("Probe Set's noisy images' template size:", noisyprobe.shape) #This line prints the template size of the noisy images of Probe Set.
dnoisy = probe_csv_results(noisyprobe, 2) #This line uses the probe_csv_results() function to store the noisy images' score matrix, histogram, and decidability index value.
print("Probe Set's profile images' template size:",profileprobe.shape) #This line prints the template size of the profile images of Probe Set.
dprofile = probe_csv_results(profileprobe, 3) #This line uses the probe_csv_results() function to store the profile images' score matrix, histogram, and decidability index value.
print("Probe Set's blurry images' template size:", blurryprobe.shape) #This line prints the template size of the blurry images of Probe Set.
dblurry = probe_csv_results(blurryprobe, 4) #This line uses the probe_csv_results() function to store the blurry images' score matrix, histogram, and decidability index value.
print("Probe Set's occluded images' template size:",occludedprobe.shape) #This line prints the template size of the occluded images of Probe Set.
doccluded = probe_csv_results(occludedprobe, 5) #This line uses the probe_csv_results() function to store the occluded images' score matrix, histogram, and decidability index value.
decidabilityindexarray = [dnoisy, dprofile, dblurry, doccluded] #This line stores the decidability index values of these 4 Probe Conditions in a list.
print("Lowest d' value:", min(decidabilityindexarray)) #This line finds the lowest decidability index value of this list.
endtime6 = time.time() #This line records the end time of the Probe Conditions.
print("Total Probe Condition runtime =", (endtime6 - starttime6), "sec or", ((endtime6 - starttime6)/60), "min") #This line prints the total execution time of the Probe Conditions.
