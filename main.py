import sys
#from pandas.tests.io.excel.test_openpyxl import openpyxl
from flight import *
from plane import *
#from selenium import webdriver
import pandas as pd
import numpy as numpy
#from bs4 import BeautifulSoup
import csv


#variables
Planes = []
Flights = []
count = 0
numbOfPlanes = 6
#code


######################################################################################

df = pd.read_excel (r'C:\Users\User\Desktop\havelsan\Bookings.xlsx', sheet_name='Bookings')
df = df.fillna('')
#sys.stdout = open("test.txt", "w")
numpy.set_printoptions(threshold=sys.maxsize)
print(df)

airports = df['AIRPORT'] # print out only the airports in the dataset
print(airports)
#########################################################################################

print("\n")


with open('flight.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1 :
            tempFlight = flight(row[0],row[1],row[2],row[3],row[4])
            Flights.append(tempFlight)

i = 0
while i < count - 1:
    print(Flights[i].name , Flights[i].leavingPlace , Flights[i].leavingTime , Flights[i].arrivalPlace , Flights[i].arrivalTime)
    i = i + 1

def assign_flight(plane,flight):
    plane.flight = flight
    plane.avilable = 0
pass

def create_randomPlanes(planeCount):
    i = 0
    while i < 3:
        p = plane("p"+str(i),0,None,"ANK")
        Planes.append(p)
        i = i + 1
    while i < 7:
        p = plane("p"+str(i),0,None,"IST")
        Planes.append(p)
        i = i + 1
    while i < 10:
        p = plane("p"+str(i),0,None,"NY")
        Planes.append(p)
        i = i + 1

pass





def schedule_flights(args,args2):
        for a in args:
            print(a.name)
        for a in args2:
            print(a.name)
pass



create_randomPlanes(10)
schedule_flights(Flights,Planes)





