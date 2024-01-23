import csv

adhd_negative = "ADHD_negative"                                     # Constant string for negative classifier
adhd_positive = "ADHD_positive"                                     # Constant string for positive classifier
filePath1 = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\DoctorsNotesNumeric.csv'                               # Positives data
filePath2 = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\DoctorsNotesControlNumeric.csv'                        # Negatives Data
fileToWritePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\DoctorsNotesData.csv'                            # Synthesis file in which we will combine the positives and the negatives

csvPositivesList = []                                               # The buffer for positives data: The buffers are used in order to remove patient names and add the classifier field.
csvNegativesList = []                                               # The buffer for negatives data
    
def main():
    # Open the file which will hold both data types: positives and negative (note that newline parameter should be '' to avoid extra newline chars):
    with open(fileToWritePath,'a',newline='',encoding='UTF-8') as FileWritten1:
        #***************************************** APPENDING THE POSITIVES *************************************************************************
        # Open the positives data which will be written first:
        with open(filePath1,'r',newline='',encoding='UTF-8') as FileRead1:
            # Initialize the reader object for csv
            csvReader = csv.reader(FileRead1)
            # Iterate over the rows of the csv file:
            for row in csvReader:
                # Append the current row to the buffer for positives:
                csvPositivesList.append(row)
            # Itearate over the rows of the positives buffer:
            for row in csvPositivesList:
                # Replace the last field which is the name of the patient for confidentiality and replace it with the positive classifier:
                row[-1]=adhd_positive
            # The buffer for positives is now complete and ready to be written to the synthesis file
            # Initialize the writer object for csv for synthesis file
            csvWriter = csv.writer(FileWritten1)
            # Write all rows in one call by utilizing csv library:
            csvWriter.writerows(csvPositivesList)
        #***************************************** APPENDING THE NEGATIVES *************************************************************************
        # Open the negatives data which will be written after having appended the positives:
        with open(filePath2,'r',newline='',encoding='UTF-8') as FileRead2:
            # Initialize the reader object for csv
            csvReader = csv.reader(FileRead2)
            # Iterate over the rows of the csv file:
            for row in csvReader:
                # Append the current row to the buffer for negatives:
                csvNegativesList.append(row)
            # Itearate over the rows of the negatives buffer:
            for row in csvNegativesList:
                # Replace the last field which is the name of the patient for confidentiality and replace it with the positive classifier:
                row[-1]=adhd_negative
            # The buffer for negatives is now complete and ready to be written to the synthesis file
            # Initialize the writer object for csv for synthesis file
            csvWriter = csv.writer(FileWritten1)
            # Write all rows in one call by utilizing csv library:
            csvWriter.writerows(csvNegativesList)

if __name__ == "__main__":
    main()