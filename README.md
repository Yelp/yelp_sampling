yelp_sampling
======================


Pyspark implementation of [Scalable Simple Random Sampling Algorithm](http://proceedings.mlr.press/v28/meng13a.pdf).


# Getting started

## SRS sampling in pyspark

```python
from yelp_sampling.scalable_srs import scalable_srs

rdd = sc.textFile(<path>)
sampled_rdd = scalable_srs(rdd, {
  'train': 10000000,
  'validation': 1000000,
  'test': 5000000,
})
```


## License

yelp_sampling is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

## References
* [Scalable Simple Random Sampling and Stratified Sampling](http://proceedings.mlr.press/v28/meng13a.pdf)
* [Scalable Simple Random Sampling Algorithms](https://www.slideshare.net/xrmeng/scalable-simple-random-sampling-algorithms)
* [spark-sampling scala](https://github.com/mengxr/spark-sampling) 

