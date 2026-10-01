import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
import cv2




class CDEvaluator():

    def __init__(self, args, dataloader, dataset_name='unknown'):

        self.dataloader = dataloader
        self.dataset_name = dataset_name
        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, f'log_test_{dataset_name}.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)
        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        self.G_pred_c = None

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)
        pred_dir = os.path.join(self.vis_dir, 'pred', self.dataset_name)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        self.pred_subdir = pred_dir

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis
    def _visualize_pred_c(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self, ckpt_name=None):

        running_acc = self._update_metric()

        m = len(self.dataloader)
        name = self.batch['name'][0]
        if np.mod(self.batch_id, 100) == 1 and self.batch_id != 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' % \
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)
        #  save the image

        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        vis_pred = utils.make_numpy_grid(self._visualize_pred())
        vis_gt = utils.make_numpy_grid(self.batch['L'])

        vis_pred = np.clip(vis_pred, a_min=0.0, a_max=1.0)
        vis_gt = np.clip(vis_gt, a_min=0.0, a_max=1.0)

        H, W, C = vis_pred.shape
        vis_conf = np.zeros_like(vis_pred)

        pred_gray = vis_pred[:, :, 0]
        gt_gray = vis_gt[:, :, 0]

        mask_red = (pred_gray > gt_gray)
        mask_green = (pred_gray < gt_gray)
        mask_equal = (pred_gray == gt_gray)

        vis_conf[mask_red] = np.array([1.0, 0.0, 0.0])
        vis_conf[mask_green] = np.array([0.0, 1.0, 0.0])
        vis_conf[mask_equal] = vis_pred[mask_equal]

        vis = np.concatenate([
            vis_input,
            vis_input2,
            vis_pred,
            vis_gt,
            vis_conf
        ], axis=0)

        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        pred_file_name = os.path.join(
            self.vis_dir,
            'pred',
            self.dataset_name,
            name
        )
        plt.imsave(pred_file_name, vis)

        if self.G_pred_c is not None:
            vis_pred_c = utils.make_numpy_grid(self._visualize_pred_c(self.G_pred_c))
            vis_pred_c = np.clip(vis_pred_c, a_min=0.0, a_max=1.0)

            vis_conf_c = np.zeros_like(vis_pred_c)
            pred_c_gray = vis_pred_c[:, :, 0]
            gt_gray = vis_gt[:, :, 0]  # gt 不变
            mask_red_c = (pred_c_gray > gt_gray)
            mask_green_c = (pred_c_gray < gt_gray)
            mask_equal_c = (pred_c_gray == gt_gray)
            vis_conf_c[mask_red_c] = np.array([1.0, 0.0, 0.0])
            vis_conf_c[mask_green_c] = np.array([0.0, 1.0, 0.0])
            vis_conf_c[mask_equal_c] = vis_pred_c[mask_equal_c]

            vis_c = np.concatenate([
                vis_input,
                vis_input2,
                vis_pred_c,
                vis_gt,
                vis_conf_c
            ], axis=0)
            vis_c = np.clip(vis_c, a_min=0.0, a_max=1.0)

            base, ext = os.path.splitext(name)
            pred_file_name_c = os.path.join(
                self.vis_dir,
                'pred',
                self.dataset_name,
                f"{base}_c{ext}"
            )
            plt.imsave(pred_file_name_c, vis_c)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass_dp(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)


    def eval_models_dp(self,ckpt_name='last_ckpt.pt'):

        self._load_checkpoint(ckpt_name)
        ckpt = str(ckpt_name)

        self.logger.write(f'========== Begin evaluation for checkpoint: {ckpt_name} ==========\n')

        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass_dp(batch)
            self._collect_running_batch_states(ckpt)

        self._collect_epoch_states()

        self.logger.write('\n')





