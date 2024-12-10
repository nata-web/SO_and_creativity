class container():
  def __init__(self,copyFrom=None):
    if copyFrom is not None:
      for option in dir(copyFrom):
        if option[:2]!='__':
          setattr(self,option,getattr(copyFrom,option))

class plotOptions(container):
  def __init__(self,copyFrom=None):
    self.dump=False
    self.saveFigures=False
    super().__init__(copyFrom)

class calcOptions(container):
  def __init__(self,copyFrom=None):
    self.unWeightedEnergies=False
    self.startState=None
    super().__init__(copyFrom)