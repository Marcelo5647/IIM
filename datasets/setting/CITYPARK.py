from easydict import EasyDict as edict


# init
__C_CITYPARK = edict()

cfg_data = __C_CITYPARK

__C_CITYPARK.TRAIN_SIZE = (576,768)
__C_CITYPARK.DATA_PATH = ''
__C_CITYPARK.TRAIN_LST = f'{__C_CITYPARK.DATA_PATH}train_data/img_list_train.txt'
__C_CITYPARK.VAL_LST =  f'{__C_CITYPARK.DATA_PATH}train_data/img_list_test.txt'
__C_CITYPARK.VAL4EVAL = f'{__C_CITYPARK.DATA_PATH}val_gt_loc.txt'

__C_CITYPARK.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932],
                 [0.278580576181, 0.26925137639, 0.27156367898])

__C_CITYPARK.LABEL_FACTOR = 1
__C_CITYPARK.LOG_PARA = 1.

__C_CITYPARK.RESUME_MODEL = ''#model path
__C_CITYPARK.TRAIN_BATCH_SIZE = 1 #imgs

__C_CITYPARK.VAL_BATCH_SIZE = 1 # must be 1