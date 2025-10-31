from easydict import EasyDict as edict

config = edict()

# Network settings - use r100 to match your pretrained model
config.network = "r100"
config.resume = False  # Set to True if resuming from checkpoint
config.output = "output/ndtwin_r100"  # Output directory
config.embedding_size = 512

# Margin loss parameters
config.margin_list = (1.0, 0.5, 0.0)
config.interclass_filtering_threshold = 0

# Training settings
config.fp16 = True  # Match your pretrained model
config.batch_size = 8  # Adjust based on your GPU memory (start smaller)
config.optimizer = "sgd"
config.lr = 0.01  # Lower learning rate for fine-tuning
config.momentum = 0.9
config.weight_decay = 5e-4
config.pretrained_path = "/kaggle/input/nd-twin-448-train/ms1mv3_arcface_r100_fp16.pth"

# Partial FC sampling (1.0 means use all classes)
config.sample_rate = 1.0

# Dataset settings - UPDATE THESE!
config.rec = "/kaggle/input/nd-twin-448-train/ND_TWIN_448_TRAIN"  # Path to your dataset folder or .rec file
config.num_classes = 377  # Count of unique people
config.num_image = 6734  # Total number of images
config.num_epoch = 20
config.warmup_epoch = 0

# Validation (optional - comment out if you don't have validation sets)
config.val_targets = []

# Training settings
config.verbose = 2000
config.frequent = 10
config.dali = False
config.gradient_acc = 1
config.seed = 2048
config.num_workers = 2

# WandB logging (optional)
config.using_wandb = False