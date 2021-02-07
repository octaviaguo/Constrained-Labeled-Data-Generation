import os
import random
import config
from model_wrapper import create_model
from pytorch_lightning.callbacks.base import Callback
import torch
import pytorch_lightning as pl
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
parser.add_argument('--load_epoch', type=int, default=-1)
parser.add_argument('--model', type=str, default='')
parser.add_argument('--pydev', type=bool, default=False, help='enable pydev debug.')
parser.add_argument('--pydev_port', type=int, default=5678)
parser.add_argument('--gpu_count', type=int, default=1)
parser.add_argument('--gpu_index', type=str, default='')
parser.add_argument('--debug_with_small_data', type=bool, default=False)

args = parser.parse_args()
config.init_config(args.config)

def get_weights_save_path(epoch, save=False):
    model_type = config.items['model']['type']
    data_type = config.items['data']['type']
    save_dir = config.items['data']['save_dir']
    noise = config.items['training']['noise']
    path_with_noise = os.path.join(
            save_dir,
            model_type.replace('/','-') + '_' + data_type + '_noise' + str(noise) +'_epoch_%d' % (epoch) + '.ckpt')
    if noise > 0.0  and (save or os.path.exists(path_with_noise)):
        path = path_with_noise
    else:
        path = os.path.join(
            save_dir,
            model_type.replace('/','-') + '_' + data_type + '_epoch_%d' % (epoch) + '.ckpt')
    if save:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    return path


class MyCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, trainer, pl_module):
        #print('==================on_epoch_end', trainer)
        #print(config.items)
        if pl_module.device.index == trainer.data_parallel_device_ids[0]:
            print('on_epoch_end:', trainer.current_epoch)
            if (trainer.current_epoch%config.items['management']['checkpoint_freq'])==0:
                torch.save(
                    pl_module.model.state_dict(),
                    get_weights_save_path(trainer.current_epoch, save=True)
                )
        #print('\n\non_epoch_end%d\n\n'%(trainer.current_epoch))

    '''
    def on_batch_end(self, trainer, pl_module):
        #print('on_batch_end', self.batch_count)
        #self.batch_count += 1
    '''

    def on_train_start(self, trainer, pl_module):
        print('=======================load state', pl_module.device.index)
        trainer.current_epoch=0
        if args.load_epoch>=0:
            load_path = get_weights_save_path(args.load_epoch)
        elif len(args.model)>0:
            load_path = args.model
        else:
            load_path = None

        if load_path is not None:
            print("\n\nLoad the previous model from: " + load_path)
            print("\n\n")
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
            pl_module.model.load_state_dict(state_dict)
            if args.load_epoch>=0:
                trainer.current_epoch = args.load_epoch+1


def create_trainer():

    max_epochs = config.get_max_epochs()
    acc_batch = config.get_accumulate_batches()

    my_callback = MyCallback()

    if len(args.gpu_index)>0:
        dev_id = args.gpu_index.split(',')
        gpus = [int(i) for i in dev_id]
    else:
        gpus = args.gpu_count
    trainer = pl.Trainer(gpus=gpus,
                         max_epochs=max_epochs,
                         check_val_every_n_epoch=1,
                         profiler=True,
                         accumulate_grad_batches=acc_batch,
                         checkpoint_callback=None,
                         callbacks = [my_callback],
                         progress_bar_refresh_rate=50,
                         resume_from_checkpoint=None)
    #                     distributed_backend='dp')

    return trainer


def train(x_train, x_val, x_test):
    model = create_model(x_train, x_val, x_test)
    trainer = create_trainer()
    trainer.fit(model)


if __name__ == "__main__":
    #config.init_config(args.config)

    src=config.items['data']['src']
    trg=config.items['data']['trg']

    train_lines = []
    #print(src, trg)
    f_src = open(src, 'r')
    f_trg = open(trg, 'r')
    for sline,tline in zip(f_src, f_trg):
        train_lines.append((sline, tline))

    src=config.items['data']['test_src']
    trg=config.items['data']['test_trg']

    test_lines = []
    f_src = open(src, 'r')
    f_trg = open(trg, 'r')
    for sline,tline in zip(f_src, f_trg):
        test_lines.append((sline, tline))

    x_train = train_lines
    x_test = test_lines
    random.shuffle(x_train)

    if args.debug_with_small_data: #for debugging
        x_val = x_train[:48]
        x_train = x_train[5000:5096]
    else:
        #x_val = x_train[:5000]
        #x_train = x_train[5000:]
        x_val = x_test

    if args.pydev:
        import pydevd
        pydevd.settrace("localhost", port=args.pydev_port) 

    train(x_train, x_val, x_test)
