import math
import sys
from tkinter.ttk import Combobox
from tkinter import ttk
from tkinter.ttk import *

from PIL import ImageTk
from pandas.tests.io.excel.test_openpyxl import openpyxl
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from flight import *
from plane import *
from selenium import webdriver
import pandas as pd
import numpy as numpy
import fileinput
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import tkinter as tk
from tkinter import *
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO
try:
    import tkinter as tk                # python 3
    from tkinter import font as tkfont  # python 3
except ImportError:
    import Tkinter as tk     # python 2
    import tkFont as tkfont  # python 2




# code
####################################################################################
# variables
Planes = []
Flights = []
printOut = []
Days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
count = 0


print("\n")



"""
numpy.random.seed(0)
arrayTemp = numpy.random.normal(size = 60, loc = 60, scale =30)
df = pd.read_csv("test.csv")
delayCol = 0

while delayCol < 60:
    df.loc[delayCol, 'delay amount'] = arrayTemp[delayCol]
    delayCol = delayCol + 1

df.to_csv("test.csv", index=False)


numpy.random.seed(0)
arrayTemp2 = numpy.random.normal(size = 60, loc = 10000, scale =5000)

df = pd.read_csv("test.csv")
rentCol = 0

while rentCol < 60:
    df.loc[rentCol, 'rental price'] = arrayTemp2[rentCol]
    rentCol = rentCol + 1

df.to_csv("test.csv", index=False)
"""

count = 0
with open('flight.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            tempFlight = flight(row[0], row[2], row[1], row[3], row[4], None, row[5], row[6])
            Flights.append(tempFlight)



def change_to_hours(number):
    temp = numpy.fix(number)
    temp2 = number - temp
    temp3 = temp2
    if temp2 >= 0.6:
        temp3 = temp2 - 0.6
        print(temp3)
        number = temp - 1 + temp3
    print(number)
pass

def assign_flight(plane, flight):
    plane.flight = flight
    flight.plane = plane
    plane.place = flight.arrivalPlace
    plane.available = 0
    Planes.remove(plane)
    Planes.append(plane)
pass


def free_plane(plane, flight):
    plane.place = flight.leavingPlace
    flight.plane = None
    plane.availableHour = 2 + float(flight.arrivalTime)
    if(plane.availableHour >= 24):
        plane.availableHour = plane.availableHour - 24
        plane.availableHour = round(float(plane.availableHour),2)

    plane.available = 1

pass


def create_randomPlanes(planeCount):
    i = 0
    while i < planeCount:
        p = plane("p" + str(i), 1, None, "IST", 0,"IST",0)
        Planes.append(p)
        i = i + 1
pass


def schedule_flights(args, args2):

    for d in Days:
        currentDate = d
        hour = 0
        counter = 0
        time = 0
        tempString = f"{currentDate}"
        printOut.append(tempString)
        while hour < 24:
            minute = 0
            while minute < 60:
                minute = minute + 1
                if minute == 60:
                    break
                counter = counter + 1  # end of one minute
                if minute < 10:
                    time = hour + (minute * (0.1))
                else:
                    time = hour + (minute * (0.01))

                time = round(float(time), 2)

                for a in args:
                    if round(float(a.leavingTime),2) == time and a.leavingDay == currentDate:
                        check = 0
                        for x in args2:
                            if x.available == 1 and x.availableHour <= time and x.place == a.leavingPlace:
                                assign_flight(x, a)
                                check = 1
                                tempString = f"time: {time} | flight name: {x.flight.name} | plane name: {x.name}  | day: {a.leavingDay} |  FLIGHT IS ASSIGNED"
                                printOut.append(tempString)
                                break
                        if check == 0:
                            for x in args2:
                                if x.available == 1 and x.availableHour <= time:
                                    assign_flight(x, a)
                                    tempString = f"time: {time} | flight name: {x.flight.name} | plane name: {x.name}  | day: {a.leavingDay} |  FLIGHT IS ASSIGNED check == 0"
                                    printOut.append(tempString)
                                    break

                    if round(float(a.arrivalTime),2) == time and a.arrivalDay == currentDate:
                        for x in args2:
                            if x.flight == a:
                                free_plane(x, a)
                                tempString = f"time: {time} | flight name: {x.flight.name} | plane name: {x.name}  | day: {a.arrivalDay} |  PLANE IS FREE"
                                printOut.append(tempString)
                                break
            hour = hour + 1
        tempString = "-----------------------------------------------------------------------------------------------------------------"
        printOut.append(tempString)
pass

data = pd.read_csv("test.csv")
# load dataset

col_names = ['place','backup plane','delay amount','rental price','label']
# load dataset

test = pd.read_csv("test.csv")

test.columns = col_names


feature_cols = ['place','backup plane','delay amount','rental price']
X = test[feature_cols] # Features
y = test.label # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = DecisionTreeClassifier(criterion="gini",min_samples_split=10,max_depth=3)
clf = clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_test)
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('test.png')
Image(graph.create_png())




""""
elastic=ElasticNet(alpha=0.5, l1_ratio=0.1, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None,).fit(X_train,y_train)
y_pred = elastic.predict(X_test)
score = elastic.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
      .format(score, mse, numpy.sqrt(mse)))

x_ax = range(len(X_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()

x_ax = range(len(X_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


def n_alpha(args):
    pass


elastic_cv=ElasticNetCV(alphas=alphas, cv=5)
model = elastic_cv.fit(X_train, y_train)
print(model.alpha_)
print(model.intercept_)

ypred = model.predict(X_test)
score = model.score(X_test, y_test)
mse = mean_squared_error(y_test, ypred)
print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
      .format(score, mse, numpy.sqrt(mse)))

x_ax = range(len(X_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()
"""""


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(StartPage)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

class StartPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master,bg="red4")
        greeting1_label = tk.Label(
            master=self,
            text="Welcome to the system, please go to the main page",
            font=("Courier", 20, "italic"),
            fg="red4",
            bg="light gray",
            width=200,
            height=3
        )
        greeting1_label.pack()
        button = tk.Button(self, text="Main Page",fg="white",bg="DodgerBlue4", width= 10,height=2,font=("Courier",16,"italic"),
                  command=lambda: master.switch_frame(PageTwo)).pack()

def combine_funcs(*funcs):
    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)
    return combined_func

class PageOne(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        schedule_flights(Flights, Planes)


        frame3 = tk.Frame(master=self, width=100,height=300, bg="red4")

        frame4 = tk.Frame(master=self, width=200, height=600, bg="red4")
        greeting1_label = tk.Label(
            master=frame3,
            text="Scheduled flights of the week",
            font=("Courier", 20, "italic"),
            fg="black",
            bg="light gray",
            width=200,
            height=3
        )
        greeting1_label.pack()

        scrollbar = Scrollbar(frame3)
        scrollbar.pack(side=RIGHT, fill=Y)

        listbox = Listbox(frame3, yscrollcommand=scrollbar.set)
        [listbox.insert(END, row) for row in printOut]
        listbox.config(font="Courier", fg="blue")
        listbox.pack(fill=BOTH, expand=1)
        scrollbar.config(command=listbox.yview)


        problem_label = tk.Label(
            master=frame4,
            text="If there is a problem with the plane, please choose the options below. ",
            font=("Courier", 16, "italic"),
            fg="white",
            bg="red4",
        )
        problem_label.place(x=30, y=60)
        plane_names = []
        for i in Planes:
            plane_names.append(i.name)

        plane_label = tk.Label(
            master=frame4,
            text="Plane name:",
            font=("Courier", 16),
            fg="white",
            bg="red4",
        )

        plane_label.place(x=60, y=120)
        cb = ttk.Combobox(frame4, values=plane_names)
        cb.place(x=60, y=150)

        delay_label = tk.Label(
            master=frame4,
            text="Delay Amount (in hours):",
            font=("Courier", 16),
            fg="white",
            bg="red4",
        )
        delay_label.place(x=300, y=120)
        delay_box = tk.Text(frame4, height=1, width=30)
        delay_box.place(x=300, y=150)
        back_button =tk.Button(master=frame4, text="Back to main page", command=lambda:master.switch_frame(PageTwo), height = 2,
                                    width=30,fg="white", bg="DodgerBlue4", font="Courier")
        back_button.place(x=1050, y=280)
        frame3.pack(fill=tk.BOTH, expand=True)
        frame4.pack(fill=tk.BOTH, expand=True)


class PageTwo(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        frame1 = tk.Frame(master=self, width=200, height=200, bg="white")


        greeting_label = tk.Label(
            master=frame1,
            text="Welcome to the simulation!!",
            font=("Courier", 20, "italic"),
            fg="black",
            bg="light gray",
            width=200,
            height=3
        )
        greeting_label.pack()

        frame2 = tk.Frame(master=self, width=20, height=800, bg="red4")
        plane_box = Entry(frame2)
        plane_box.place(x=600, y=150)

        def retrieve_input():
            inputValue = int(plane_box.get())
            create_randomPlanes(inputValue)

        plane_label = tk.Label(
            master=frame2,
            text="Please enter the number of planes.",
            font=("Courier", 14, "italic"),
            fg="lightblue1",
            bg="red4",
            width=100,
            height=3
        )
        plane_label.place(x=200,y=50)


        button_add_planes = tk.Button(master=frame2, text="Ok", command=lambda: retrieve_input(), height=2,
                                      width=10, fg="red4",bg="lightblue1", font="Courier")

        button_add_planes.place(x=750, y=120)
        button_start = tk.Button(master=frame1, text="Back to start page", command=lambda:master.switch_frame(StartPage), height=3,
                                    width=50, fg ="white",bg="DodgerBlue4", font="Courier").pack()

        button_schedule = tk.Button(master=frame1, text="See scheduled flights!", command=lambda: master.switch_frame(PageOne), height=3,
                                    width=50,fg="white", bg="DodgerBlue4", font="Courier")

        frame1.pack(fill=tk.BOTH, side=tk.TOP, expand=True)
        button_schedule.place(relx=0.5, rely=0.5, anchor=CENTER)
        button_schedule.pack(pady=30)

        frame2.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)







if __name__ == "__main__":
    app = App()
    app.mainloop()


