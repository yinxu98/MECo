_base_ = ['./classifier_opensarship_test.py']

checkpoints = dict(
    folders=[
        '../work_dirs/multiembedding_opensarship_pretrain',
    ],
    index=[
        '0020.pth',
        '0040.pth',
        '0060.pth',
        '0080.pth',
        '0100.pth',
    ],
)
