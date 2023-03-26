[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# CacheTest

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The code and data in this repository are a snapshot of the code and data
that were used in the research reported on in the paper 
"Monte Carlo Methods for Economic Capital" (https://doi.org/10.1287/ijoc.2019.0000). 
The snapshot is based on 
[this SHA](https://github.com/tkralphs/JoCTemplate/commit/f7f30c63adbcb0811e5a133e1def696b74f3ba15) 
in the development repository. 

**Important: This code is being developed on an on-going basis at 
https://github.com/tkralphs/JoCTemplate. Please go there if you would like to
get a more recent version or would like support**

## Cite

To cite the contents of this respository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2019.0000

https://doi.org/10.1287/ijoc.2019.0000.cd

Below is the BibTex for citing this snapshot of the respoitory.

```
@article{CacheTest,
  author =        {L. Yajuan, N. Marvin, K. Zachary},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Monte Carlo Methods for Economic Capital}},
  year =          {2023},
  doi =           {10.1287/ijoc.2019.0000.cd},
  url =           {https://github.com/INFORMSJoC/2021.0261.git},
}  
```



## Description

This repository contains code for the paper, "Monte Carlo Methods for Economic Capital," which has been accepted for publication in the INFORMS Journal on Computing.

This repository is divided into two folders. The folder PCRM contains the code for portfolioi credit risk model in the paper. The other folder contains code for i.i.d. model.

PCRM codes are written in python, and i.i.d. model codes are written in matlab.

## Results

Figure 1 in the paper shows the results of the multiplication test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/mult-test.png)

Figure 2 in the paper shows the results of the sum test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/sum-test.png)

## Replicating

To replicate the results in [Figure 1](results/mult-test), do either

```
make mult-test
```
or
```
python test.py mult
```
To replicate the results in [Figure 2](results/sum-test), do either

```
make sum-test
```
or
```
python test.py sum
```

## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site](https://github.com/tkralphs/JoCTemplate).

## Support

For support in using this software, submit an
[issue](https://github.com/tkralphs/JoCTemplate/issues/new).
