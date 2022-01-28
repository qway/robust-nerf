_base_ = '../default.py'

expname = 'first'
basedir = './logs/nerfies'

data = dict(
    datadir='./data/nerfies/first',
    dataset_type='nerfies',
    white_bkgd=False,
)

