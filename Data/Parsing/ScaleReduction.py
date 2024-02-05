import csv
import numpy as np

filePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\ConnersTeacherData.csv'
fileToWritePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\ConnersTeacherDataScaledDown.csv'

csvConnersList = []

csvScaledDownList = []

def scaleDown(filePath, fileToWritePath):
    #***************************************** READ THE CONNERS DATA *************************************************************************
    # Open the positives data which will be written first:
    with open(filePath,'r',newline='',encoding='UTF-8') as FileRead:
        # Initialize the reader object for csv
        csvReader = csv.reader(FileRead)
        # Iterate over the rows of the csv file:
        for row in csvReader:
            # Append the current row to the buffer for the first file:
            csvConnersList.append(row)
    #***************************************** SCALE DOWN THE CONNERS DATA ********************************************************************
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
            else:
                currentSubList.append(item)
        #print(currentSubList)
        csvScaledDownList.append(currentSubList)
    #***************************************** EMPTY THE FILE TO BE WRITTEN IF ALREADY EXISTS *********************************************************
    with open(fileToWritePath,'w',newline='',encoding='UTF-8') as fileToBeModified:
        fileToBeModified.write("")

    #***************************************** WRITE TO THE FILE ******************************************************************************
    # Open the file which will hold the combined data (note that newline parameter should be '' to avoid extra newline chars):
    with open(fileToWritePath,'a',newline='',encoding='UTF-8') as FileWritten:
        # Initialize the writer object for csv for synthesis file
            csvWriter = csv.writer(FileWritten)
            # Write all rows in one call by utilizing csv library:
            csvWriter.writerows(csvScaledDownList)

def main():
    scaleDown(filePath, fileToWritePath)

if __name__ == "__main__":
    main()