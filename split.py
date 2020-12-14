import sys
import os
from sklearn.model_selection import train_test_split

with open(sys.argv[1], 'r') as f:
    data = f.readlines()

print(sys.argv[1], sys.argv[2])
print(int(round(len(data)*.6)))
def p(d,f): return int(round(len(d)*f))
#train = data[:p(data, .6)]
#test = data[p(data, .6):p(data, .8)]
#dev = data[p(data, .8):]
train, test = train_test_split( data, test_size=.4)
test, dev = train_test_split( test, test_size=.5)

assert len(data) == len(train) + len(test) + len(dev)

with open(os.path.join(sys.argv[2], 'user_info.train'), 'w') as f: 
    f.writelines(train)

with open(os.path.join(sys.argv[2], 'user_info.test'), 'w') as f: 
    f.writelines(test)

with open(os.path.join(sys.argv[2], 'user_info.dev'), 'w') as f: 
    f.writelines(dev)
