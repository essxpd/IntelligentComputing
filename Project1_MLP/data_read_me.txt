Three clouds
------------
3 classes, 2 dimensional, 300 instances

data format (per row):
[class label, d1, d2]


Wine
----
3 classes, 13 dimensional, 178 instances

data format (per row):
[class label, d1, d2, ..., d13]


Semeion (handwritten digits 0-9)
--------------------------------
10 classes, 256 dimensional, 1593 instances

data format (per row):
[d1, d2, ..., d256, label_0, label_1, ..., label_9]

To plot instance 441 (a 16x16 binary image) and get its label in Matlab:

load('semeion.data');
imshow(reshape(semeion(441,1:256),16,16)');
label = find( semeion(441,257:end) ) - 1;

