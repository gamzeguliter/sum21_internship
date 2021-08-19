from tkinter import ttk
from tkinter.ttk import *
from sklearn.model_selection import train_test_split
from flight import *
from plane import *
import pandas as pd
import tkinter as tk
import csv
from tkinter import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import six
import sys
import matplotlib.pyplot as plot

sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO

try:
    import tkinter as tk
    from tkinter import font as tkfont
except ImportError:
    import Tkinter as tk
    import tkFont as tkfont
import numpy as numpy
import fileinput
from datetime import datetime
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn import tree
import pydotplus
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
import math
import sys
from tkinter.ttk import Combobox
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

#################################  CODE  ################################################################################

# global variables
Planes = []  # array of planes in the simulation
Flights = []  # aray of flights in the simulation
printOut = []
Days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
count = 0
totalFlightHours = 0
flightPercentage = 0

# below commented code is to assign random values to the parameters in test.csv by using Gaussian Distribution
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

# Read flight data from flight csv
count = 0
with open('flight.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            tempFlight = flight(row[0], row[2], row[1], row[3], row[4], None, row[5], row[6])
            Flights.append(tempFlight)


# Function to assign flights to planes
def assign_flight(plane, flight):
    plane.flight = flight
    flight.plane = plane
    plane.place = flight.arrivalPlace
    plane.flightHour = plane.flightHour + addToTotalFlightHour(float(flight.leavingTime), float(flight.arrivalTime))
    plane.available = 0
    Planes.remove(plane)
    Planes.append(plane)


pass


# Free planes from assigned flight
def free_plane(plane, flight):
    plane.place = flight.leavingPlace
    flight.plane = None
    plane.availableHour = 2 + float(flight.arrivalTime)
    if (plane.availableHour >= 24):
        plane.availableHour = plane.availableHour - 24
        plane.availableHour = round(float(plane.availableHour), 2)

    plane.available = 1


pass


# creates number of planes with the given count
def create_randomPlanes(planeCount):
    i = 0
    while i < planeCount:
        p = plane("p" + str(i), 1, None, "IST", 0, "IST", 0, i)
        Planes.append(p)
        i = i + 1
    return i


pass


# helper function to find the time difference between leaving and arrival time of a plane in MINUTES
def addToTotalFlightHour(leave, arrive):
    number_dec = leave % 1
    number_dec2 = arrive % 1
    number_dec = round(float(number_dec), 2)
    number_dec2 = round(float(number_dec2), 2)
    number = int(leave)
    number2 = int(arrive)
    minutes = number * 60 + number_dec * 100
    minutes2 = number2 * 60 + number_dec2 * 100

    if (arrive > leave):
        totalFlightHours = (minutes2 - minutes)

    else:
        totalFlightHours = (24 * 60) - minutes + minutes2

    return totalFlightHours


# Function which performs 7 days of simulation
"""""
Notes:
-Assigns planes to flights
-Frees planes when the flight is over
"""""


def schedule_flights(args, args2):
    for d in Days:  # days
        currentDate = d
        hour = 0
        counter = 0
        time = 0
        tempString = f"{currentDate}"

        printOut.append(tempString)
        while hour < 24:  # hours
            minute = 0
            while minute < 60:  # minutes
                minute = minute + 1
                if minute == 60:
                    break
                counter = counter + 1  # end of one minute
                if minute < 10:
                    time = hour + (minute * (0.1))
                else:
                    time = hour + (minute * (0.01))

                time = round(float(time), 2)  # current time

                for a in args:  # args = flights , looks for the leaving time of flights and assigns planes to them
                    if round(float(a.leavingTime), 2) == time and a.leavingDay == currentDate:
                        check = 0
                        for x in args2:  # args2 = planes , looks for the free planes and assigns them to flights
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
                    # free planes when the arrival time of their flight has come
                    if round(float(a.arrivalTime), 2) == time and a.arrivalDay == currentDate:
                        for x in args2:
                            if x.flight == a:
                                free_plane(x, a)
                                tempString = f"time: {time} | flight name: {x.flight.name} | plane name: {x.name}  | day: {a.arrivalDay} |  PLANE IS FREE"
                                printOut.append(tempString)
                                break
            hour = hour + 1
        tempString = "-----------------------------------------------------------------------------------------------------------------"
        printOut.append(tempString)

    # Print out flight amount of each plane in the simulation
    for p in Planes:
        print(p.name, " => ", p.flightHour, "minutes of flight")

    print("------------------------------------------------------------------------------------------------------------------")


    # calculates total avg percentage of the flights
    calculatePercentage = 0
    for p in Planes:
        print(p.name, "---flight percentage---", (int(p.flightHour)) * 100 / totalFlightHours)
        calculatePercentage = calculatePercentage + (int(p.flightHour)) * 100 / totalFlightHours

    return calculatePercentage


pass

# below code is to try elasticNet algorithm for ML model.
# the model failed with low accurency rate

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


# APP class is the window of the UI
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.wm_title("Flight Simulation")
        self._frame = None
        self.switch_frame(StartPage)

    # below function is used to change the view o the window, in order to change pages
    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

"""""
Class of the Starting page of the simulation
"""""
class StartPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg="red4")
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
        button = tk.Button(self, text="Main Page", fg="white", bg="DodgerBlue4", width=10, height=2,
                           font=("Courier", 16, "italic"),
                           command=lambda: master.switch_frame(PageTwo)).pack()


""""
Class of the Flight schedule page of the simulation
Notes: 
- The Page1 frame consists of 2 other sub-frames (Frame3 and Frame4)
"""""
class PageOne(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        flightPercentage = schedule_flights(Flights, Planes)
        planeCount = 0

        # calculate avg percentage of fligts for each plane
        for p in Planes:
            planeCount = planeCount + 1
        print("Average percentage of flight : ", float(round(flightPercentage / planeCount, 2)))

        # create sub-frames
        frame3 = tk.Frame(master=self, width=100, height=300, bg="red4")
        frame4 = tk.Frame(master=self, width=200, height=600, bg="red4")

        # welcoming label
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
            font=("Courier", 12),
            fg="white",
            bg="red4",
        )
        plane_label.place(x=60, y=120)
        cb = ttk.Combobox(frame4, values=plane_names)
        cb.place(x=60, y=150)

        delay_label = tk.Label(
            master=frame4,
            text="Delay Amount (in minutes):",
            font=("Courier", 12),
            fg="white",
            bg="red4",
        )
        delay_label.place(x=300, y=120)
        delay_box = tk.Text(frame4, height=1, width=30)
        delay_box.place(x=300, y=150)

        back_button = tk.Button(master=frame4, text="Back to main page", command=lambda: master.switch_frame(PageTwo),
                                height=2,
                                width=20, fg="white", bg="DodgerBlue4", font="Courier")
        back_button.place(x=1200, y=350)

        backupPlaneLabel = tk.Label(
            master=frame4,
            text="Are there any back up planes?",
            font=("Courier", 12),
            fg="white",
            bg="red4",
        )
        backupPlaneLabel.place(x=60, y=200)
        cb2 = ttk.Combobox(frame4, values=["yes", "no"])
        cb2.place(x=60, y=230)

        backupPlaneLabel_closest = tk.Label(
            master=frame4,
            text="Are there any back up planes in close areas?",
            font=("Courier", 12),
            fg="white",
            bg="red4",
        )
        backupPlaneLabel_closest.place(x=400, y=200)
        cb3 = ttk.Combobox(frame4, values=["yes", "no"])
        cb3.place(x=400, y=230)

        rent_label = tk.Label(
            master=frame4,
            text="Enter the rental cost.",
            font=("Courier", 12),
            fg="white",
            bg="red4",
        )
        rent_label.place(x=60, y=280)
        rent_box = tk.Text(frame4, height=1, width=15)
        rent_box.place(x=60, y=310)

        # below function is using Decision Tree in order to decide how to solve the problem occured in scheduling (ML)
        def fix_Schedule():
            popup = tk.Tk()
            popup.wm_title("Suggested Solution")

            # get user inputs to predict solution
            delayAmount = int(delay_box.get("1.0", 'end-1c'))
            rentAmount = int(rent_box.get("1.0", 'end-1c'))

            if (cb2.get() == "yes"):
                back_up_plane = 1
            else:
                back_up_plane = 0

            if (cb3.get() == "yes"):
                back_up_plane2 = 1
            else:
                back_up_plane2 = 0

            # prepare prediction data
            dataTotest = [back_up_plane2, back_up_plane, delayAmount, rentAmount]

            # decision tree implementation

            data = pd.read_csv("test.csv")  # load data to decision tree
            col_names = ['place', 'backup plane', 'delay amount', 'rental price', 'label']
            test = pd.read_csv("test.csv")
            test.columns = col_names
            feature_cols = ['place', 'backup plane', 'delay amount', 'rental price']
            X = test[feature_cols]
            y = test.label  # target label
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                random_state=42)  # split data to test/train pars

            clf = DecisionTreeClassifier(criterion="gini", min_samples_split=10, max_depth=3)
            clf = clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            print("######################################################################################", "\n")
            print("Accuracy of the decision tree based on the training and test data:",
                  metrics.accuracy_score(y_test, y_pred))
            print("######################################################################################", "\n")

            arrayf = [dataTotest]
            y_pred2 = clf.predict(arrayf)
            print(y_pred2)
            solutionLabel = y_pred2[0]
            solution = " hi"
            if solutionLabel == 0:
                solution = "The problem requires no solution"
            if solutionLabel == 1:
                solution = " Offered solution is to use backup plane"
            if solutionLabel == 2:
                solution = " Offered solution is to rent another firm's plane "
            if solutionLabel == 3:
                solution = " Offered solution is to assign another plane from a close place "

            label = tk.Label(popup, text=solution, bg="red4", fg="lightblue1", font=("Courier"))
            label.pack(side="top", fill=BOTH, pady=10)
            button_okey = tk.Button(popup, text="Okay!", bg="DodgerBlue4", fg="lightblue1", font=("Courier"),
                                    command=popup.destroy).pack()

            # Note: graphviz is not working in company pc
            # below code is used to visualize decision tree
            """""
            dot_data = StringIO()
            export_graphviz(clf, out_file=dot_data,
                            filled=True, rounded=True,
                            special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3'])
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            graph.write_png('test.png')
            Image(graph.create_png())

            B1.pack()
            popup.mainloop
            """""

        fix_button = tk.Button(master=frame4, text="Fix Schedule",
                               command=lambda: fix_Schedule(),
                               height=1,
                               width=18, fg="white", bg="DodgerBlue4", font="Courier")
        fix_button.place(x=60, y=350)

        frame3.pack(fill=tk.BOTH, expand=True)
        frame4.pack(fill=tk.BOTH, expand=True)


""""
Class of the Main Page of the simulation where the user indicates the number of planes in the simulation
Notes: 
- The Page2 frame consists of 2 other sub-frames (Frame2 and Frame1)
"""""
class PageTwo(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        # create sub-frames
        frame1 = tk.Frame(master=self, width=200, height=200, bg="white")
        frame2 = tk.Frame(master=self, width=20, height=800, bg="red4")

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

        plane_box = Entry(frame2)
        plane_box.place(x=600, y=150)

        #below function is used to retrive user input in button press
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
        plane_label.place(x=200, y=50)

        button_add_planes = tk.Button(master=frame2, text="Ok", command=lambda: retrieve_input(), height=2,
                                      width=10, fg="red4", bg="lightblue1", font="Courier")

        button_add_planes.place(x=750, y=120)
        button_start = tk.Button(master=frame1, text="Back to start page",
                                 command=lambda: master.switch_frame(StartPage), height=3,
                                 width=50, fg="white", bg="DodgerBlue4", font="Courier").pack()

        button_schedule = tk.Button(master=frame1, text="See scheduled flights!",
                                    command=lambda: master.switch_frame(PageOne), height=3,
                                    width=50, fg="white", bg="DodgerBlue4", font="Courier")

        frame1.pack(fill=tk.BOTH, side=tk.TOP, expand=True)
        button_schedule.place(relx=0.5, rely=0.5, anchor=CENTER)
        button_schedule.pack(pady=30)

        frame2.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)


if __name__ == "__main__":

    count = 1

    #Calculate total flight hour in the simulation by reading flight.csv file
    with open('flight.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if count != 1:
                totalFlightHours = totalFlightHours + addToTotalFlightHour(float(row[1]), float(row[2]))
            count = count + 1

    finalResultHour = totalFlightHours / 60
    finalResultMin = totalFlightHours % 60

    print("TOTAL FLIGHT AMOUNT OF THE WEEK : ", totalFlightHours, "minutes")
    print("TOTAL FLIGHT AMOUNT OF THE WEEK : ", float(round(finalResultHour)), "hours", finalResultMin, "minutes")

    # below are the codes to draw a plot to see the corelation between the number of planes vs the avg time of fligts
    x1 = [2, 5, 10, 15, 20]
    y1 = [22.49, 16.33, 9.63, 6.73, 5.77]

    plot.plot(x1, y1, label="avg flight time")

    plot.xlabel('Num. of planes')
    plot.ylabel('Avg flight time')
    plot.title('Number of Planes Used vs Avg Flight Time Plot')
    plot.show()

    # start running the simulation
    app = App()
    app.mainloop()
