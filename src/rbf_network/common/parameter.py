class Parameter(object):
    def __init__(self):
        self.__old_values = []
        self.__value = 0

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        self.__old_values.append(self.__value)
        self.__value = value

    def clear(self):
        del self.__old_values[:]

    def undo(self):
        if(len(self.__old_values) != 0):
            self.__value = self.__old_values.pop()

