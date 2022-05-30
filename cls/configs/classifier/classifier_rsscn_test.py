_base_ = [
    './classifier_test.py',
    '../_base_/datasets/rsscn.py',
]

model = dict(classifier=dict(dim_out=7))
