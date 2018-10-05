# Social Network Formation

This is the public codes and data for the "An Interpretable Approach for Social NetworkFormation Among Heterogeneous Agents" (to appear in Nature Communications). 

## Prerequisite
* Python (Python 3 recommended)
* tensorflow
* numpy
* networkx


## Download
Please directly download the zip file or git clone the repository in command line:

```
$ git clone
```

## Data description

### Network data

Sample data used in our paper are in the "data" folder. Individuals must be indexed by 0, 1, 2, ... The number of agents are inferred by the max index plus 1. Because we only consider undirected graphs, for each input pair, \(i, j\), we add \(j, i\) automatically. 

For example, below are the first 10 lines of karate network.

```
0 1
0 2
0 3
0 4
0 5
0 6
0 7
0 8
0 10
0 11
```

For the detail for other networks, please refer to the paper.


### Attribute data

Attribute data are stored in "data_attribute" folder. We present the meaning of each column as below:

#### Karate
* Faction (0 or 1)

#### Trade
* Continent (the first four columns): 1 indicates Africa, America, Asia/Pacific and Europe, respectively
* Economic complexity index (ECI) 
* GDP

#### Synthetic
* Type
* x
* y

#### MobileD
* Manager (1) or subordinate (9)

#### Movie
* Director (0) or cast members (1)
* Male (0) or female (1)

#### Andorra
* Phone type (Apple, Samsung, etc)
* District (0 to 6)
* SCDR (cellular usage)


## Code description

### learning.py 

The code to learn the endowment vectors.

Please type in command line for further instruction. By default we run on the toy network (Karate club). 

```
$ python learning.py --help
```

For example, you may use the command below for trade network

```
$ python learning.py --input data/trade.txt --output output/trade --dimension 4 --bst 2
```
### dynamics.py

Please revise W, b, c, beta, f to the output that you obtain.

```
W = np.loadtxt('output/log_karate_4_W_2.txt')
B = np.loadtxt('output/log_karate_4_B_2.txt')
C = np.loadtxt('output/log_karate_4_C_2.txt')
beta = np.loadtxt('output/log_karate_4_beta_2.txt')
f = open('data/karate.txt')
```
