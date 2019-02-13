# [Data challenge]

## Presentation

This is the code for my various submissions at the data challenge **[TO BE REVEALED]**

## TODO

* `benchmark.py`, `bag_of_words.py` - set best parameters to est before passing it to classify, to ensure CV is only performed on some parameters (avoid `key: [value]` in `cv_params`)
* `benchmark.py`, `bag_of_words.py` - Fix preprocessing issues that cause a mess between x_tr and x_tr.values
* Preprocessing the entire training set before cross-validation is slightly cheating - maybe implement a new wrapper class and use pipelines?