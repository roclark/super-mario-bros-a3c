class Range:
    def __init__(self, start, end):
        self._start = start
        self._end = end

    def __eq__(self, input_num):
        return self._start <= input_num <= self._end
