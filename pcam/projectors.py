import numpy as np
import math
import torch
import torch.nn.functional as F

from dataset_pcam import PCamDataModule

def create_targets(create_projector, num_targets: int, datamodule: PCamDataModule):
    
    data_sample = torch.cat([batch[0] for batch in iter(datamodule.train_dataloader(samples_per_class=500))], 0)
    return __select_targets([__initialize_projector(create_projector()) for _ in range(num_targets * 10)], num_targets, data_sample)

# reference: https://github.com/layer6ai-labs/lfr/blob/main/utils/model_utils.py
def __initialize_projector(projector: torch.nn.Module):
    # In the authors' code, they apply the dropout before the beta initialization.
    # In my understanding, the beta initialization would override the dropout if executed in that order.
    # create_targets execute the initialization first, to make sure that dropout weights won't come back.
    projector.apply(__initialize_layer_with_beta)
    projector.apply(__randomly_set_to_zero)
    projector.apply(__regularize_parameters)
    return projector

def __initialize_layer_with_beta(m):
    with torch.no_grad():
        if type(m) == torch.nn.Conv2d:
            # sample weights intialization from beta distribution
            beta_dist = torch.distributions.beta.Beta(torch.tensor([0.5]), 
                        torch.tensor([0.5]))
            weight_size = m.weight.size()
            # scale to [-1,1]
            random_weights = 2*beta_dist.sample(weight_size)-1
            m.weight.data = random_weights.view(weight_size)

def __randomly_set_to_zero(m):
    with torch.no_grad():
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            n = m.weight.numel()
            drop_num = int(round(n*0.4))
            indices = torch.randperm(n)[:drop_num]
            m.weight = m.weight.contiguous()  
            m.weight.flatten()[indices] = 0

def __regularize_parameters(m):
    with torch.no_grad():
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            for param in m.parameters():
                param.data = param.data/torch.norm(param.data)

# reference: https://github.com/layer6ai-labs/lfr/blob/main/ssl_models/lfr.py
def __select_targets(target_encoders, num_targets, sample_data):
    '''
    select num_targets number of encoders out of target_encoders
    ''' 
    with torch.no_grad():
        sims = []
        for t in target_encoders:
            # (bs, dim)
            rep = t(sample_data)
            if rep.shape[0] > 1000: 
                rep = rep[np.random.RandomState(seed=42).permutation(np.arange(rep.shape[0]))[:1000]]
            rep_normalized = F.normalize(rep, dim=1)
            # (bs, bs) cosine similarity
            sim = rep_normalized @ rep_normalized.T
            sims.append(sim.view(-1))
        # N, bs^2
        sims = torch.stack(sims)
        sims_normalized = F.normalize(sims, dim=1)
        # N,N
        sims_targets = sims_normalized @ sims_normalized.T
        result = __dpp(sims_targets.cpu().numpy(), num_targets)
    return [target_encoders[idx] for idx in result]

# reference: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py
def __dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items