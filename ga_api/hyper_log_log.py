from copy import copy
import math

class HyperLogLog:
    def __init__(self, arg=None):
        if arg is None: # zero
            self._rmem = bytearray(2048)
        elif isinstance(arg, str): # HyperLogLog Literal
            if arg.startswith('h:'):
                self._rmem = bytearray.fromhex(arg[2:])
            else:
                self._rmem = bytearray.fromhex(arg)
        elif type(arg) == bytearray: # register mem
            self._rmem = copy(arg)
        else:
            raise Exception('HyperLogLog construct error')
        pass

    def __add__(self, other):
        return HyperLogLog(bytearray(max(a,b) for a, b in zip(self._rmem, other._rmem)))

    def __radd__(self, other):
        return HyperLogLog(bytearray(max(a,b) for a, b in zip(self._rmem, other._rmem)))

    def __iadd__(self, other):
        for i in range(len(self._rmem)):
            self._rmem[i] = max(self._rmem[i], other._rmem[i])
        return self

    def __str__(self):
        return f'{self.value()}'

    def __repr__(self):
        return f'h:{self._rmem.hex()}'

    def insert(self, hval):
        hval &= 0x7FFFF
        pos = hval >> 8 # 0..2047
        val = hval & 0xFF # 0..255
        self._rmem[pos] = max(val, self._rmem[pos])

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
