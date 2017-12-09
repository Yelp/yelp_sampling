yelp_sampling
======================


Pyspark implementation of [Scalable Simple Random Sampling Algorithm](http://proceedings.mlr.press/v28/meng13a.pdf).


# Getting started

## SRS sampling in pyspark

```python
sc.addPyFile('yelp_sampling/srs_sampling.py')
from srs_sampling import sample_train_test_split 

sample_train_test_split(sc, 
train_size, 
test_size, 
's3 location of samples', 
's3 bucket to dump training samples', 
's3 bucket to dump test samples')
```


## License

yelp_sampling is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

## References
* [Scalable Simple Random Sampling and Stratified Sampling](http://proceedings.mlr.press/v28/meng13a.pdf)
* [Scalable Simple Random Sampling Algorithms](https://www.slideshare.net/xrmeng/scalable-simple-random-sampling-algorithms)
* [spark-sampling scala](https://github.com/mengxr/spark-sampling) 

