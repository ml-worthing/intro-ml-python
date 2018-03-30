from collections import namedtuple


Point = namedtuple('Point', ['x', 'y'], verbose=True)

p1 = Point(x=11, y=22)
p2 = Point(x=11, y=22)
p3 = Point(x=55, y=456)

print(p1)       #Point(x=11, y=22)
print(p1 == p2) #True
print(p2 == p3) #False
print(p1 + p2)  #(11, 22, 11, 22)

