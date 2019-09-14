# Timeflux ML plugin

This plugin provides timeflux nodes and meta-nodes for real time machine learning on time series.

## Installation

First, make sure that [Timeflux is installed](https://github.com/timeflux/timeflux).

You can then install this plugin in the ``timeflux`` environment:

```
$ conda activate timeflux
$ pip install git+https://github.com/timeflux/timeflux_ml
```

## Modules

### fit
This module contains nodes to create and fit pipeline of transforms with eventually a final estimator.

### transform
This module contains nodes to transform data thanks to a previously fitted pipeline.

### predict
This module contains nodes to predict the label of data thanks to a previously fitted pipeline.
