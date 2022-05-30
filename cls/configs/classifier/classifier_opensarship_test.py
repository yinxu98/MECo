_base_ = [
    './classifier_test.py',
    '../_base_/datasets/opensarship.py',
]

model = dict(classifier=dict(dim_out=3))
