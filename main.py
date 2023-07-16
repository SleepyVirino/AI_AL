import random

epoch = 10000000
count = 0

for i in range(epoch):
    x = random.uniform(0,1)
    y = random.uniform(0,1)
    z = random.uniform(0,1)
    if z*z > x*y:
        count += 1

print(count/epoch)
print(5/9)