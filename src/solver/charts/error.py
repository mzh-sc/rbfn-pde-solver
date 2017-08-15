class Error(object):
    def __init__(self, figure, position):
        self.__axes = figure.add_subplot(position)
        self.__axes.grid(True)
        self.__axes.set_ylabel('Error')

        self.__errors = []

    def add_error(self, error):
        self.__errors.append(error)

    def update(self):
        self.__axes.plot(range(len(self.__errors)), self.__errors, 'b-')
