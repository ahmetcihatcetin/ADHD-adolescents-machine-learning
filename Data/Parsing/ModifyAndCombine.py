import csv
import numpy as np

adhd_negative = "ADHD_negative"                                     # Constant string for negative classifier
adhd_positive = "ADHD_positive"                                     # Constant string for positive classifier

filePath1 = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\ConnersParentData.csv'                                  # Conners Data
filePath2 = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\DoctorsNotesData.csv'                                   # Doctors Notes Data
fileToWritePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\CombinedDataBinaryConnersParentAndDoctorsNotes.csv'     # Combined Data

csvConnersList = []             # The buffer for the rows of the first list

csvModifiedList = []

csvDoctorsNotesList = []        # The buffer for the rows of the second list

csvFinalList = []               # The final list which will be written to the file specified by fileToWritePath

def combine(filePath1, filePath2, fileToWritePath):         # Combines the two different patient data into one such that for each row the data for an invidiual match.
                                                            # filePath1 is the filepath for the Conners data, filePath2 is the filepath for the Doctors Notes data 
    #***************************************** READ THE CONNERS DATA *************************************************************************
    # Open the positives data which will be written first:
    with open(filePath1,'r',newline='',encoding='UTF-8') as FileRead1:
        # Initialize the reader object for csv
        csvReader = csv.reader(FileRead1)
        # Iterate over the rows of the csv file:
        for row in csvReader:
            # Append the current row to the buffer for the first file:
            csvConnersList.append(row)
        # Itearate over the rows of the Conners data buffer:
        for row in csvConnersList:
            # Replace the last field which is the name of the patient for confidentiality and replace it with the positive classifier:
            row.pop()
    #***************************************** MODIFY THE CONNERS DATA ************************************************************************
    # Since the doctor's notes data is binary, the scale of the Conner's data need to be reduced to binary as well for the fair usage of the both data by the ML model.
    for row in csvConnersList:
        currentSubList = []
        for item in row:
            #print(item)
            if item == '0':
                currentSubList.append('0')
            elif item == '1':
                currentSubList.append('0')
            elif item == '2':
                currentSubList.append('1')
            elif item == '3':
                currentSubList.append('1')
        #print(currentSubList)
        csvModifiedList.append(currentSubList)
            
    #***************************************** READ THE DOCTORS NOTES *************************************************************************
    # Open the doctors note data which will be read and then written into its seperate buffers:
    with open(filePath2,'r',newline='',encoding='UTF-8') as FileRead2:
        # Initialize the reader object for csv
        csvReader = csv.reader(FileRead2)
        # Iterate over the rows of the csv file:
        for row in csvReader:
            # Append the current row to the buffer for negatives:
            csvDoctorsNotesList.append(row)
    #***************************************** COMBINE THE DATA *******************************************************************************
    for i in range(0,len(csvModifiedList)):
        csvFinalList.append(np.append(csvModifiedList[i], csvDoctorsNotesList[i]))

    #***************************************** EMPTY THE FILE TO BE WRITTEN IF ALREADY EXISTS *********************************************************
    with open(fileToWritePath,'w',newline='',encoding='UTF-8') as fileToBeModified:
        fileToBeModified.write("")

    #***************************************** WRITE TO THE FILE ******************************************************************************
    # Open the file which will hold the combined data (note that newline parameter should be '' to avoid extra newline chars):
    with open(fileToWritePath,'a',newline='',encoding='UTF-8') as FileWritten1:
        # Initialize the writer object for csv for synthesis file
            csvWriter = csv.writer(FileWritten1)
            # Write all rows in one call by utilizing csv library:
            csvWriter.writerows(csvFinalList)

def main():
    combine(filePath1, filePath2, fileToWritePath)

if __name__ == "__main__":
    main()