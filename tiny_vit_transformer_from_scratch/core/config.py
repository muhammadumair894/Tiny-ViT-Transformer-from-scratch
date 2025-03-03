import torch

class VitConfig:
    def __init__(self):  
        # Adjusted for ImageNet 1K
        self.batch_size = 128    # was 16
        self.n_embd = 256        # you could go higher (e.g., 768) for a standard ViT-B/16
        self.n_block = 12
        self.h_size = 32
        self.p_size = 16
        self.c_dim = 3
        self.im_size = 224       # typical ImageNet resolution
        self.n_class = 1000      # ImageNet 1K
        self.d_rate = 0.0        # can be 0.1 during fine-tuning
        self.bias = False
        self.h_dim = 6 * self.n_embd  # hidden dimension in MLP blocks

    def __repr__(self):
        return (
            f"<VitConfig batch_size={self.batch_size}, embedding_dim={self.n_embd}, "
            f"num_blocks={self.n_block}, hidden_size={self.h_dim}, patch_size={self.p_size}, "
            f"channel_dim={self.c_dim}, im_size={self.im_size}, num_classes={self.n_class}, "
            f"dropout_rate={self.d_rate}, use_bias={self.bias}, head_dim={self.h_size}>"
        )


class Config:
    def __init__(self, batch_size, im_size, n_class) -> None:
        # Checkpoint paths
        self.save_ckpt_path = "./vit_chpts.pth"
        self.load_chpt_path = "./vit_chpts.pth"

        # Optimizer settings
        self.lr_rate = 2e-4
        self.w_decay = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.amsgrad = False

        # Training set size: full ImageNet is ~1,281,167 images, here we approximate
        self.train_size = 1280000 

        # We want ~300 epochs total
        # Option 1: Hardcode `num_epochs=300` and compute `total_iters`:
        self.batch_size = batch_size
        self.num_epochs = 300
        # Steps per epoch = train_size // batch_size
        steps_per_epoch = self.train_size // self.batch_size
        self.total_iters = self.num_epochs * steps_per_epoch

        # Option 2: Keep total_iters fixed and let num_epochs be derived:
        # self.total_iters = 3000000
        # self.batch_size = batch_size
        # self.num_epochs = (self.total_iters * self.batch_size) // self.train_size

        # Warmup (can reduce these if you don’t want 300 epochs of warmup)
        self.warmup_epochs = 5
        self.warmup_iters = self.warmup_epochs * steps_per_epoch

        # Other training details
        self.min_lr = 1e-6
        self.label_smoothing = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Unused or placeholder class labels for prior tasks; can remove or ignore
        self.classes = ["Placeholder"] * n_class

        # Data loading
        self.valid_size = 0.2
        self.im_size = im_size
        self.num_workers = 8
        self.pin_memory = True
        self.shuffle = True

        # Point this to your ImageNet directory
        self.data_dir = "/kaggle/input/imagenetmini-1000/imagenet-mini"

        # Turn off dataset-size-limiting
        self.max_img_cls = None
        self.max_cls = None
        self.is_balanced = False

    def __repr__(self):
        return (
            f"<Config save_ckpt_path={self.save_ckpt_path}, load_chpt_path={self.load_chpt_path}, "
            f"lr_rate={self.lr_rate}, w_decay={self.w_decay}, beta1={self.beta1}, beta2={self.beta2}, "
            f"eps={self.eps}, amsgrad={self.amsgrad}, total_iters={self.total_iters}, "
            f"num_epochs={self.num_epochs}, warmup_epochs={self.warmup_epochs}, device={self.device}, "
            f"warmup_iters={self.warmup_iters}, min_lr={self.min_lr}, label_smoothing={self.label_smoothing}, "
            f"train_size={self.train_size}, batch_size={self.batch_size}>"
        )
