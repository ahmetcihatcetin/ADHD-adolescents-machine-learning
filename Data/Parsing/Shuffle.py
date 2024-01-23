import random
import csv

buffer = []

def shuffle(filePath):
    with open(filePath,'r',newline='',encoding='UTF-8') as fileToBeModified:
        csvReader = csv.reader(fileToBeModified)
        for row in csvReader:
            buffer.append(row)
    with open(filePath,'w',newline='',encoding='UTF-8') as fileToBeModified:
        fileToBeModified.write("")
    random.shuffle(buffer)
    with open(filePath,'a',newline='',encoding='UTF-8') as fileToBeWritten:
        csvWriter = csv.writer(fileToBeWritten)
        csvWriter.writerows(buffer)

def main():
    shuffle(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Parsing\DoctorsNotesData.csv' )

if __name__ == "__main__":
    main()