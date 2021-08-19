"""""
Class of the object flight
@param arrivalTime = arrival time of the flight to arrivalPlace
@param leavingTime = leaving time of the flight to leavingPlace
@para plane        = assigned plane to the flight 
"""""


class flight:
    def __init__(self,name,arrivalTime,leavingTime,arrivalPlace,leavingPlace,plane,leavingDay,arrivalDay):
        self.name = name
        self.plane = plane
        self.arrivalDay = arrivalDay
        self.leavingDay = leavingDay
        self.arrivalTime = arrivalTime
        self.leavingTime = leavingTime
        self.arrivalPlace = arrivalPlace
        self.leavingPlace = leavingPlace

