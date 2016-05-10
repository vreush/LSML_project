
FTRL in C++

####Compilation:
g++ -std=c++0x -o lr t.cpp


####Input file format:
target,feature1:value1,feature2:value2,...,featureN:valueN
target in (0,1)


####How to use:
./lr train.vw test.vw out.csv batch_size alpha beta lambda1 lambda2

- example: ./lr train.vw test.vw out.csv 5000000 0.1 1.0 1.0 1.0

####HW1 data converter for testing:
conv_hw1.py
