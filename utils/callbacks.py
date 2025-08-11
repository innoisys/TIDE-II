import os.path

import tensorflow as tf


class VisualizeCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_interval=1, func=lambda model, epoch: None):
        super(VisualizeCallback, self).__init__()
        self.func = func
        self.epoch_interval = epoch_interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_interval == 0 and epoch > 0:
            self.func(self.model, epoch)


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, vae, path, epoch_interval=1, restore_training=False, restore_path=None):
        super(CheckpointCallback, self).__init__()
        self.epoch_interval = epoch_interval
        self.path = path
        self.vae = vae

        self.ckpt = tf.train.Checkpoint(vae=vae,
                                        vae_optimizer=vae.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.ckpt,
                                                       directory=self.path,
                                                       max_to_keep=None)
        self.restore_training = restore_training
        self.restore_path = restore_path
        self._saved = False

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_interval == 0 and epoch > 0:
            self.ckpt_manager.save(checkpoint_number=epoch)

    def on_train_begin(self, logs=None):
        if self.restore_training:
            if self.restore_path is None:
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint).except_partial()
                print("Resume training from checkpoint ", self.ckpt_manager.latest_checkpoint, "\n")
            else:
                self.ckpt.restore(self.restore_path)
                print("resume training from checkpoint ", self.restore_path, "\n")

    def on_train_end(self, logs=None):
        weights_path = os.path.join(self.path, "trained-vae")
        self.ckpt.save(file_prefix=weights_path)

