# code adpated from: https://github.com/jarrycyx/UNN
import os
import itertools
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
import tqdm
from einops import rearrange
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from trainer import Trainer, prepare_inputs
from utils.misc import mkdir


def plot_matrix(name, mat, log_dir, log_step, vmin=None, vmax=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if len(mat.shape) == 3:
        mat = np.max(mat, axis=-1)
    n, m = mat.shape

    # Show Discovered Graph (Probability)
    sub_cg = plot_causal_matrix(
        mat,
        # figsize=[1.5*n, 1*n],
        figsize=[6, 4],
        show_text=False,
        vmin=vmin, vmax=vmax)
    sub_cg.savefig(f"{log_dir}/{name}_{log_step}.pdf")


def generate_indices(input_step, pred_step, t_length, block_size=None):
    if block_size is None:
        block_size = t_length
        
    offsets_in_block = np.arange(input_step, block_size-pred_step+1)
    assert t_length % block_size == 0, "t_length % block_size != 0"
    random_t_list = []
    for block_start in range(0, t_length, block_size):
        random_t_list += (offsets_in_block + block_start).tolist()
    
    np.random.shuffle(random_t_list)
    return random_t_list


def batch_generater(data, observ_mask, bs, n_nodes, input_step, pred_step, block_size=None):
    
    t, n_nodes = data.shape
    first_sample_t = input_step
    random_t_list = generate_indices(input_step, pred_step, t_length=t, block_size=block_size)

    for batch_i in range(len(random_t_list) // bs):
        x = torch.zeros([bs, n_nodes, input_step]).cuda()
        y = torch.zeros([bs, n_nodes, pred_step]).cuda()
        t = torch.zeros([bs]).cuda().long()
        mask_x = torch.zeros([bs, n_nodes, input_step]).cuda()
        mask_y = torch.zeros([bs, n_nodes, pred_step]).cuda()
        for data_i in range(bs):
            data_t = random_t_list.pop()
            x[data_i, :, :] = rearrange(data[data_t-input_step : data_t, :], "t n -> n t")
            y[data_i, :, :] = rearrange(data[data_t : data_t+pred_step, :], "t n -> n t")
            t[data_i] = data_t
            mask_x[data_i, :, :] = rearrange(observ_mask[data_t-input_step : data_t, :], "t n -> n t")
            mask_y[data_i, :, :] = rearrange(observ_mask[data_t:data_t+pred_step, :], "t n -> n t")

        yield x, y, t, mask_x, mask_y


def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def plot_causal_matrix(cmtx, class_names=None, figsize=None, vmin=None, vmax=None, show_text=True, cmap="magma"):
    """
    A function to create a colored and labeled causal matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): causal matrix.
        num_classes (int): total number of nodes.
        class_names (Optional[list of strs]): a list of node names.
        figsize (Optional[float, float]): the figure size of the causal matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    num_classes = cmtx.shape[0]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    
    figsize[0] = 30 if figsize[0] > 30 else figsize[0]
    figsize[1] = 20 if figsize[1] > 20 else figsize[1]
    
    plt.clf()
    plt.close("all")
    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest",
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Causal matrix")
    plt.colorbar()
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names, rotation=45)
    # plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] < threshold else "black"
        if cmtx.shape[0] < 20 and show_text:
            plt.text(j, i, format(cmtx[i, j], ".2e") if cmtx[i, j] != 0 else ".",
                    horizontalalignment="center", color=color,)

    plt.tight_layout()
    plt.ylabel("Cause")
    plt.xlabel("Effect")

    return figure

    
class CUTS_PLUS_Trainer(Trainer):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.cfg = cfg
        self.cfg_model = cfg.CUTS_PLUS
        self.cfg_train = cfg.CUTS_PLUS.TRAIN

        self.data_pred_loss = nn.MSELoss()
        self.data_pred_optimizer = torch.optim.Adam(
            (param for name, param in self.model.named_parameters() if name != "GT"),
            lr=self.cfg_model.DATA_PRED.LR_DATA_START,
            weight_decay=self.cfg_model.DATA_PRED.WEIGHT_DECAY
            )
        
        
        if "every" in self.cfg_model.FILL_POLICY:
            lr_schedule_length = int(self.cfg_model.FILL_POLICY.split("_")[-1])
        else:
            lr_schedule_length = self.cfg_model.SOLVER.MAX_EPOCH
            
        gamma = (self.cfg_model.DATA_PRED.LR_DATA_END / self.cfg_model.DATA_PRED.LR_DATA_START) ** (1 / lr_schedule_length)
        self.data_pred_scheduler = torch.optim.lr_scheduler.StepLR(
            self.data_pred_optimizer, step_size=1, gamma=gamma)
        
        self.n_groups = self.cfg_model.N_GROUPS
        print("n_groups: ", self.n_groups)
        if self.cfg_model.GROUP_POLICY == "None":
            self.cfg_model.GROUP_POLICY = None

        end_tau, start_tau = self.cfg_model.GRAPH_DISCOV.END_TAU, self.cfg_model.GRAPH_DISCOV.START_TAU
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.cfg_model.SOLVER.MAX_EPOCH)
        self.gumbel_tau = start_tau
        self.start_tau = start_tau
        
        end_lmd, start_lmd = self.cfg_model.GRAPH_DISCOV.LAMBDA_S_END, self.cfg_model.GRAPH_DISCOV.LAMBDA_S_START
        self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.cfg_model.SOLVER.MAX_EPOCH)
        self.lambda_s = start_lmd
        
        self.best_auc = 0
    
    def set_graph_optimizer(self, epoch=None):
        if epoch == None:
            epoch = 0
        
        gamma = (self.cfg_model.GRAPH_DISCOV.LR_GRAPH_END / self.cfg_model.GRAPH_DISCOV.LR_GRAPH_START) ** (1 / self.cfg_model.SOLVER.MAX_EPOCH)
        self.graph_optimizer = torch.optim.Adam([self.model.GT], lr=self.cfg_model.GRAPH_DISCOV.LR_GRAPH_START * gamma ** epoch)
        self.graph_scheduler = torch.optim.lr_scheduler.StepLR(self.graph_optimizer, step_size=1, gamma=gamma)

    def latent_data_pred(self, x, y):
        
        def sample_bernoulli(sample_matrix, batch_size):
            sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
            return torch.bernoulli(sample_matrix).float()
        
        def sample_multinorm(sample_matrix, batch_size):
            sampled = torch.multinomial(sample_matrix, batch_size, replacement=True).T
            return F.one_hot(sampled).float()
            
        
        bs, n, t = x.shape
        self.model.train()
        self.data_pred_optimizer.zero_grad()
        
        GT_prob = self.model.GT
        G_prob = self.G
        
        Graph = torch.einsum("nm,ml->nl", G_prob, torch.sigmoid(GT_prob))
        graph_sampled = sample_bernoulli(Graph, self.cfg_model.TRAIN.BATCH_SIZE)
        
        y_pred = self.model(x, graph_sampled)
        y_pred = y_pred.transpose(1, 2)
        assert y.shape == y_pred.shape

        # print(y_pred.shape, y.shape, observ_mask.shape)
        loss = self.data_pred_loss(y, y_pred)
        loss.backward()
        if self.cfg_model.DATA_PRED.GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg_model.DATA_PRED.GRADIENT_CLIP_NORM)
        self.data_pred_optimizer.step()
        return y_pred, loss

    def graph_discov(self, x, y):

        def gumbel_sigmoid_sample(graph, batch_size, tau=1):
            prob = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
            logits = torch.concat([prob, (1-prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :, 0]
            return samples
        
        gn, n = self.model.GT.shape
        # self.model.eval()
        self.graph_optimizer.zero_grad()
        GT_prob = self.model.GT
        G_prob = self.G
        
        Graph = torch.einsum("nm,ml->nl", G_prob, torch.sigmoid(GT_prob))
        graph_sampled = gumbel_sigmoid_sample(Graph, self.cfg_model.TRAIN.BATCH_SIZE) 
        
        loss_sparsity = torch.linalg.norm(Graph.flatten(), ord=1) / (n * n)

        y_pred = self.model(x, graph_sampled)
        y_pred = y_pred.transpose(1, 2)
        assert y.shape == y_pred.shape
        
        loss_data = self.data_pred_loss(y, y_pred)
        loss = loss_sparsity * self.lambda_s + loss_data
        loss.backward()
        if self.cfg_model.GRAPH_DISCOV.GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_([self.model.GT], self.cfg_model.GRAPH_DISCOV.GRADIENT_CLIP_NORM)
        self.graph_optimizer.step()
        
        return loss, loss_sparsity, loss_data

    def train(self):
        
        if hasattr(self.train_loader.dataset, 'true_cm'):
            true_cm = self.train_loader.dataset.true_cm
        else:
            true_cm = None

        metric_best = self.cfg_model.TRAIN.METRIC_BEST
        
        latent_pred_step = 0
        graph_discov_step = 0
        pbar = tqdm.tqdm(total=self.cfg_model.SOLVER.MAX_EPOCH)
        auc = 0
        for epoch_i in range(self.cfg_model.SOLVER.MAX_EPOCH):
            self.G = torch.eye(self.cfg_model.N_NODES).cuda()
            self.set_graph_optimizer(epoch_i)
            
            # Data Prediction
            if hasattr(self.cfg_model, "DATA_PRED"):
                if hasattr(self.cfg_model, "BLOCK_SIZE"):
                    block_size = self.cfg_model.BLOCK_SIZE
                else:
                    block_size = None
                
                for inputs, _ in self.train_loader:
                    inputs = prepare_inputs(inputs)
                    assert self.cfg_model.INPUT_STEP + self.cfg_model.DATA_PRED.PRED_STEP == self.cfg.DATA.WIN_SIZE
                    x = inputs[:, :self.cfg_model.INPUT_STEP]
                    y = inputs[:, self.cfg_model.INPUT_STEP:]
                    
                    latent_pred_step += self.cfg_model.TRAIN.BATCH_SIZE
                    y_pred, loss = self.latent_data_pred(x, y)
                    
                    pbar.set_postfix_str(f"S1 loss={loss.item():.2f}, spr=IDLE, auc={auc:.4f}")

                current_data_pred_lr = self.data_pred_optimizer.param_groups[0]['lr']
                self.data_pred_scheduler.step()
                
                if true_cm is not None:
                    Graph = torch.sigmoid(self.model.GT).detach().cpu().numpy()
                    auc = roc_auc_score(true_cm.reshape(-1) > 0.5, Graph.reshape(-1))
                
            # Graph Discovery
            if hasattr(self.cfg_model, "GRAPH_DISCOV"):
                for inputs, _ in self.train_loader:
                    inputs = prepare_inputs(inputs)
                    assert self.cfg_model.INPUT_STEP + self.cfg_model.DATA_PRED.PRED_STEP == self.cfg.DATA.WIN_SIZE
                    x = inputs[:, :self.cfg_model.INPUT_STEP]
                    y = inputs[:, self.cfg_model.INPUT_STEP:]
                    
                    graph_discov_step += self.cfg_model.TRAIN.BATCH_SIZE
                    loss, loss_sparsity, loss_data = self.graph_discov(x, y)
                    pbar.set_postfix_str(f"S2 loss={loss_data.item():.2f}, spr={loss_sparsity.item():.2f}, auc={auc:.4f}")
                    
                self.graph_scheduler.step()
                current_graph_disconv_lr = self.graph_optimizer.param_groups[0]['lr']
                if true_cm is not None:
                    GT_prob = torch.sigmoid(self.model.GT).detach().cpu().numpy()
                    auc = roc_auc_score(true_cm.reshape(-1) > 0.5, GT_prob.reshape(-1))
                self.gumbel_tau *= self.gumbel_tau_gamma
                self.lambda_s *= self.lambda_gamma

            pbar.update(1)
            
            plot_roc = False
            
            G_prob = self.G.detach().cpu().numpy()
            GT_prob = self.model.GT.detach().cpu().numpy()
            Graph = torch.einsum("nm,ml->nl", self.G, torch.sigmoid(self.model.GT)).detach().cpu().numpy()
            
            if (epoch_i+1) % self.cfg_model.SHOW_GRAPH_EVERY == 0:
                plot_matrix("Graph", Graph, self.cfg_model.RESULT_DIR, graph_discov_step)
                np.save(os.path.join(self.cfg_model.RESULT_DIR, 'Graph.npy'), Graph)
                # plot_roc = True
            
            # Show TPR FPR AUC ROC
            if true_cm is not None:
                Graph = torch.sigmoid(self.model.GT).detach().cpu().numpy()
                auc = roc_auc_score(true_cm.reshape(-1) > 0.5, Graph.reshape(-1))
                if auc > self.best_auc:
                    self.best_auc = auc

            if (epoch_i + 1) % self.cfg_model.TRAIN.EVAL_PERIOD == 0:
                val_loss = self.eval_epoch()
                # check improvement
                is_best = self._check_improvement(val_loss, metric_best)
                # Save a checkpoint on improvement.
                if is_best:
                    with open(mkdir(self.cfg_model.RESULT_DIR) / "best_result.txt", 'w') as f:
                        f.write(f"Validation loss: {val_loss}\tEpoch: {epoch_i}")
                    print(f"[current best] Validation loss: {val_loss}\tEpoch: {epoch_i}")
                    self.save_best_model()
                    metric_best = val_loss
        
        return Graph

    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()

        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, _ in self.val_loader:
                inputs = prepare_inputs(inputs)
                x = inputs[:, :self.cfg_model.INPUT_STEP]
                y = inputs[:, self.cfg_model.INPUT_STEP:]
                Graph = torch.einsum("nm,ml->nl", self.G, torch.sigmoid(self.model.GT))
                Graph = Graph[None].expand(x.size(0), -1, -1)
                # graph_sampled = sample_bernoulli(Graph, self.cfg_model.TRAIN.BATCH_SIZE)
                y_pred = self.model(x, Graph)
                y_pred = y_pred.transpose(1, 2)
                assert y.shape == y_pred.shape
                
                loss = self.data_pred_loss(y, y_pred)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        avg_loss = total_loss / total_samples

        return avg_loss
    
    def save_best_model(self):
        print('Saving best model')
        checkpoint = {
            "model_state": self.model.state_dict(),
            "data_pred_optimizer_state": self.data_pred_optimizer.state_dict(),
            "graph_optimizer_state": self.graph_optimizer.state_dict(),
            "cfg": self.cfg.dump(),
        }
        with open(mkdir(self.cfg_model.TRAIN.CHECKPOINT_DIR) / 'checkpoint_best.pth', "wb") as f:
            torch.save(checkpoint, f)