from .config import CfgNode as CN

# ----------------------------------------------------------------------------- #
# Convention about Training / Test specific parameters
# ----------------------------------------------------------------------------- #
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# ----------------------------------------------------------------------------- #
# Config definition
# ----------------------------------------------------------------------------- #

_C = CN()
_C.SEED = 20
# ---------------------------------------------------------------------------- #
# Classification model config
# ---------------------------------------------------------------------------- #
task_mode = "Classification"
_C.DATASETS = CN()
_C.DATASETS.TRAIN = "ImageNet_train_LT"    #train dataset
_C.DATASETS.VALID = "ImageNet_valid_LT"    #valid dataset
_C.DATASETS.TEST = "ImageNet_test"      #test dataset
_C.DATASETS.INPUTE_SIZE = (224,224)

_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "densenet161"  #select classification model
_C.MODEL.NUM_CLASS = 1000    #set class num
_C.MODEL.DEVICE = "cuda:0"     #select device
_C.MODEL.JUST_EVAL = False    
_C.MODEL.PRE_WEIGHT = False    
_C.MODEL.OUTPUT_DIR = "output/"+task_mode+"/"+_C.DATASETS.TRAIN.split("_")[0]+f"_SEED{_C.SEED}"+"/"+_C.MODEL.META_ARCHITECTURE+"/"    #Path to save training log files and network weights files
_C.MODEL.PREWEIGHT = r"/home/djz/Documents/Network_weights/Confidence_calibration_under_imbalance_data/output/Classification/ImageNet_SEED20/densenet161/densenet161_2.pth"    #The path of saving the pretrain weight 
_C.MODEL.DROPOUT = False


_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "SGD"    #select optimizer，see：customKing\solver\build.py
_C.SOLVER.BATCH_SIZE = 64    #Set batch_size
_C.SOLVER.SHUFFLE = True    
_C.SOLVER.NUM_WORKS = 8    #the num workers of the Dataloader
_C.SOLVER.PIN_MEMORY = True
_C.SOLVER.IS_PARALLEL = False   #Whether to use multiple GPUs for training

_C.SOLVER.LR_SCHEDULER_NAME = "Step_Decay"     #select lr_scheduler，see：customKing\solver\build.py
_C.SOLVER.START_ITER = 0    
_C.SOLVER.MAX_EPOCH = 150    
_C.SOLVER.MAX_ITER = 480000
_C.SOLVER.BASE_LR = 0.1    
_C.SOLVER.MOMENTUM = 0.9  
_C.SOLVER.NESTEROV = False  
_C.SOLVER.WEIGHT_DECAY = 0.0001     
_C.SOLVER.GAMMA = 0.1    #if using Step_Decay，the lr after decay is BASE_LR * GAMMA
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = [240000,480000]    #Set the decay step size, which must be smaller than the training MAX_ITER.
_C.SOLVER.CLR_STEPS = 200    #if using CLR lr_scheduler, the config need to set.
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"
