from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('lr_non_native_nmt')
class NonNativeNMTSchedule(FairseqLRScheduler):
    """lr scheduler from NAACL 2019 accepted paper
    'Neural Machine Translation of Text from Non-Native Speakers'
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule'
            )
        warmup_end_lr = args.lr[0] # reaches this lr after first "warmup_updates" updates
        self.min_lr = 1.e-5 # do not use args.min_lr to avoid unexpected train ending

        # start from lr calculated as follows
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = args.encoder_embed_dim ** (-0.5) * args.warmup_updates ** (-1.5)

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = args.encoder_embed_dim ** (-0.5)

        # initial learning rate
        self.lr = args.warmup_init_lr
        args.lr_shrink = 0.0
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = max(self.min_lr, self.decay_factor * num_updates ** (-0.5))
        self.optimizer.set_lr(self.lr)
        return self.lr

