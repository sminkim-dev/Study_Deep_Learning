import numpy as np

def AND_gate(x,y):
    if x == 1 and y == 1:
        return 1
    else:
        return 0

def OR_gate(x,y):
    if x == 0 and y == 0:
        return 0
    else:
        return 1

print(AND_gate(1,1))
print(OR_gate(1,1))