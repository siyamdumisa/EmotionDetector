
class Property:
    """
       initialization function
       property functions (sets and gets)"""

    def __init__(self):
        self._xValue = None
        self._yValue = None

    # gets
    @property
    def xtest(self):
        return self._xValue

    @property
    def ytest(self):
        return self._yValue

    # setters
    @xtest.setter
    def xtest(self, v):
        self._xValue = v

    @ytest.setter
    def ytest(self, v):
        self._yValue = v

