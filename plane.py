"""""
Class of the object plane
@param available     = shows whether that plane is empty or not
@param flight        = associated flight of the plane
@param place         = is the last landed place of the plane
@param flightHour    = total flight amount of the plane in a week (in minutes)
@param hub           =  the hub where plane rests
@param availableHour = the time when the plane is ready to fly again 
@param number        = associated number of the plane
"""""
class plane:
    def __init__(self,name,available,flight,place,flightHour,hub,availableHour,number):
        self.flightHour = flightHour
        self.name = name
        self.place = place
        self.available = available
        self.flight = flight
        self.hub = hub
        self.availableHour = availableHour
        self.number = number
        pass