class KeyErrorException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
    
class TimeOutRangeException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg