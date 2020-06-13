<img src="https://github.com/mandalsubhajit/venndata/blob/master/venndata.png" width="200">


# venndata - plot Venn diagrams for arbitrary number of sets in Python

Inspired by **venn.js** package for d3js (https://github.com/benfred/venn.js/) 

## Brief description

In data science, we often require to visualize Venn diagrams for more than 3 sets. This module helps with that. Usage is very straightforward with usually pretty good results. The fineTune option should be set at 'False' if not required by the situation; it helps to get exact positions but is far slower. This module can directly read from a dataframe with membership indicating columns and calculates all overlaps itself.

**Note:** This is an approximate method because always a perfect solution does not exist. Especially in case of disjoint sets or subsets, these anomalies become evident. However, it tries to reach the best possible solution in any case.

## Getting started

### Installation

Directly from the source - clone this repo on local machine, open the shell, navigate to this directory and run:
```
python setup.py install
```
or through pip:
```
pip install venndata
```

### Documentation

**Usage**

Start by importing the modules.
```python
from venndata import venn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

Create a dummy dataframe for input -
```python
df = pd.DataFrame(np.random.choice([0,1], size = (50000, 5)), columns=list('ABCDE'))
```
Each column of this datarame should represent a set,
and the values in each column represents the membership of the observations (rows) in the corresponding
set. For example, if a particular row looks like:
```
A B C D E
1 0 0 1 0
```
it means the observation corresponding to that row belongs to sets A & D (value=1), but not to B, C or E (value=0)


Calculate the intersections between the sets. fineTune should be usually set to False, as it gives good results most of the time. If the results do not appear right, use fineTune=True and it is expected to run slowly for several minutes on an average machine.
```python
fineTune=False
labels, radii, actualOverlaps, disjointOverlaps = venn.df2areas(df, fineTune=fineTune)
```

The code can start from here also if the data calculated in the last step is directly available. Description below:

labels: the names/labels of sets as list (e.g. ```['A', 'B', 'C', 'D', 'E']```)

radii: radii of the circles representing the sets = sqrt(setsize/PI) as list (e.g. ```[89.1151690252529, 88.99721848412602, 89.32744386164458, 89.03297757367739, 89.38265957850528]```)

actualOverlaps: dictionary of all two set intersection sizes (e.g. '11000' represents intersection of A & B)
```{'11000': 12306, '10100': 12507, '10010': 12385, '10001': 12593, '01100': 12425, '01010': 12519, '01001': 12471, '00110': 12421, '00101': 12554, '00011': 12460}```

disjointOverlaps: dictionary of all mutually disjoint intersection sizes (e.g. '11000' represents intersection of A & B only
= A int B int (not C) int (not D) int (not E)). This is NOT required and can be just an empty dict {} if fineTune
is set to False.
```{'10101': 1656, '01100': 1622, '10000': 1605, '00001': 1605, '01010': 1603, '10111': 1600, '01011': 1597, '01110': 1596, '00101': 1593, '11011': 1587, '11001': 1587, '10001': 1587, '10100': 1582, '00100': 1566, '10110': 1566, '00010': 1559, '11010': 1557, '01101': 1553, '11100': 1552, '01111': 1551, '00110': 1540, '00111': 1540, '00000': 1539, '11111': 1538, '01001': 1535, '00011': 1532, '10010': 1532, '11101': 1523, '01000': 1520, '10011': 1515, '11110': 1490, '11000': 1472}```



And finally we are ready to plot the Venn diagrams.
```python
# Plot the Venn diagrams
fig, ax = venn.venn(radii, actualOverlaps, disjointOverlaps, labels=labels, labelsize='auto', cmap=None, fineTune=fineTune)
plt.savefig('venn.png', dpi=300, transparent=True)
plt.close()
```
<img src="https://github.com/mandalsubhajit/venndata/blob/master/venn.png" width="400">


## Citing **venndata**

To cite the library if you use it in scientific publications (or anywhere else, if you wish), please use the link to the GitHub repository (https://github.com/mandalsubhajit/venndata). Thank you!
