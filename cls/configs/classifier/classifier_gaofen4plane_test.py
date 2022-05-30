_base_ = [
    './classifier_test.py',
    '../_base_/datasets/gaofen4plane.py',
]

model = dict(classifier=dict(dim_out=7))
