import csv
import numpy as np

adhd_negative = "ADHD_negative"                                     # Constant string for negative classifier
adhd_positive = "ADHD_positive"                                     # Constant string for positive classifier

filePath1 = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\DoctorsNotesData.csv'
filePath2 = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\RiskFactorsData.csv'
fileToWritePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\CombinedDoctorsNotesAndRiskFactors.csv'

csvList1 = []               # The buffer for the rows of the first list

csvList2 = []               # The buffer for the rows of the second list

csvFinalList = []           # The final list which will be written to the file specified by fileToWritePath

def combine(filePath1, filePath2, fileToWritePath):         # Combines the both data for each individual row w/o any scale reduction to binary.
                                                            # filePath1 is the filepath for the Conners data, filePath2 is the filepath for the Doctors Notes data 
    #***************************************** READ THE First DATA *************************************************************************
    # Open the positives data which will be written first:
    with open(filePath1,'r',newline='',encoding='UTF-8') as FileRead1:
        # Initialize the reader object for csv
        csvReader = csv.reader(FileRead1)
        # Iterate over the rows of the csv file:
        for row in csvReader:
            # Append the current row to the buffer for the first file:
            csvList1.append(row)
        # Itearate over the rows of the Conners data buffer:
        for row in csvList1:
            # Replace the last field which is the name of the patient for confidentiality and replace it with the positive classifier:
            row.pop()
    #***************************************** READ THE SECOND DATA *************************************************************************
    # Open the second data file which will be read and then written into its seperate buffers:
    with open(filePath2,'r',newline='',encoding='UTF-8') as FileRead2:
        # Initialize the reader object for csv
        csvReader = csv.reader(FileRead2)
        # Iterate over the rows of the csv file:
        for row in csvReader:
            # Append the current row to the buffer for negatives:
            csvList2.append(row)
    #***************************************** COMBINE THE DATA *******************************************************************************
    for i in range(0,len(csvList1)):
        csvFinalList.append(np.append(csvList1[i], csvList2[i]))

    #***************************************** EMPTY THE FILE TO BE WRITTEN IF ALREADY EXISTS *********************************************************
    with open(fileToWritePath,'w',newline='',encoding='UTF-8') as fileToBeModified:
        fileToBeModified.write("")

    # Open the file which will hold the combined data (note that newline parameter should be '' to avoid extra newline chars):
    with open(fileToWritePath,'a',newline='',encoding='UTF-8') as FileWritten:
        # Initialize the writer object for csv for synthesis file
            csvWriter = csv.writer(FileWritten)
            # Write all rows in one call by utilizing csv library:
            csvWriter.writerows(csvFinalList)

def main():
    combine(filePath1, filePath2, fileToWritePath)

if __name__ == "__main__":
    main()