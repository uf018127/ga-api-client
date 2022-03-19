class HyperLogLog:
    def __init__(self, arg=None):
        if arg is None: # zero
            self._rmem = bytes(2048)
        elif isinstance(arg, str): # HyperLogLog Literal
            self._rmem = bytes.fromhex(arg[2:])
        elif type(arg) == bytes: # register mem
            self._rmem = arg
        else:
            raise Exception('HyperLogLog construct error')
        pass

    def __add__(self, other):
        return HyperLogLog(bytes(max(a,b) for a, b in zip(self._rmem, other._rmem)))

    def __str__(self):
        return f'h:{self.value()}'

    def value(self):
        N = 1 << 11
        V = 0
        S = 0
        for r in self._rmem:
            if r == 0:
                V += 1
            S += 2**(0-r)
        E = (N*N*0.7213/(1.0+1.079/N))/S
        if (E <= N*5.0/2.0 and V != 0): # small range correction
            E = N*math.log(N/V)
        elif (E >= (2**32)/30.0): # large range correction
            E = -1*(2**32)*math.log(1-E*(2**(-32)));
        return int(round(E))
