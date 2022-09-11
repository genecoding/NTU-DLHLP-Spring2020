##########################
# code by Gene, Jul 2022 #
##########################
import os
import gc
import torch 
import random
import numpy as np 
import pickle as pk
from functools import reduce
from itertools import accumulate
import matplotlib.pyplot as plt
from torch.nn import CosineSimilarity
from sklearn.decomposition import PCA

N_DATA = 1000
N_LAYER = 13
N_WORD = 128
N_FEATURE = 768
MODELS = {'pretrained': '/content/xnli-pretrained-example-data.p',
          'finetuned': '/content/xnli-finetune-example-data.p'}
eps = 1e-5


def load_data(path, device='cuda'):
    data = pk.load(open(path, 'rb'))
    input_ids = torch.tensor(np.array([data[n]['input_ids'] for n in range(N_DATA)])).to(device)
    embeddings = torch.tensor(np.array([data[n]['layer_'+str(l)] for l in range(N_LAYER) for n in range(N_DATA)])).to(device)
    embeddings = embeddings.reshape(N_LAYER, N_DATA, N_WORD, N_FEATURE)
    return input_ids, embeddings


def draw_result(results, filename):
    for model in results:
        color = 'purple' if model == 'pretrained' else 'green'
        label = 'Pretrained Model' if model == 'pretrained' else 'Finetuned Model'
        plt.plot([y.item() for y in results[model]], 'o-', color=color, label=label)
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.show()
    return


def arrange_data(input_ids, embeddings):
    """
    Arrange data by word, put embeddings of the same word together.
    """
    # mask padding, [CLS] and [SEP]
    mask = (input_ids != 0) & (input_ids != 101) & (input_ids != 102)
    # if a word appears only once in the dataset, 
    # remove it from calculating self-similarity or MEV.
    unique_w, counts = torch.unique(input_ids, return_counts=True)
    one_time_w = unique_w[counts==1]
    mask &= reduce(lambda x, y: x & y, [input_ids != w for w in one_time_w])
    valid_words_idx = mask.nonzero()

    sorted_ids, sorted_idx = input_ids[mask].sort()
    sorted_embeddings= embeddings[:, valid_words_idx[:, 0], valid_words_idx[:, 1]][:, sorted_idx]
    # (13, # of valid words, 768)
    
    return sorted_ids, sorted_embeddings
    

def release_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    return


def calculate_MEV(sorted_embeddings, idx_start_end):
    pca = PCA(n_components=0.9)
    np.seterr(invalid='ignore')  # hide pca RuntimeWarning
        
    mev_per_layer = []
    for l in range(N_LAYER):
        mev = []
        for idx in idx_start_end:
            pca.fit(sorted_embeddings[l, idx[0]:idx[1]].detach().numpy())
            mev.append(pca.singular_values_[0]**2 / (np.sum(pca.singular_values_**2) + eps))  # avoid 0/0
        mev_per_layer.append(np.mean(mev))    

    return torch.tensor(mev_per_layer)


def anisotropy(version):
    """
    There are three version options: 'self-sim', 'intra-sentence-sim' and 'MEV(BONUS)'.
    """
    if version in ['self-sim', 'intra-sentence-sim']:
        results = {}
        for model in MODELS:
            input_ids, embeddings = load_data(MODELS[model])
            
            # mask padding, [CLS] and [SEP]
            mask = (input_ids != 0) & (input_ids != 101) & (input_ids != 102)
            valid_words_idx = mask.nonzero()
            
            # sample 1000 pairs randomly
            random.seed(0)
            samples_idx = torch.tensor([random.sample(range(valid_words_idx.shape[0]), 2) for _ in range(1000)])
            w_list1_idx = valid_words_idx[samples_idx][:, 0]  # [n-th data, m-th word]
            w_list2_idx = valid_words_idx[samples_idx][:, 1]
        
            CosSim = CosineSimilarity(dim=2)
            cos_sim = CosSim(embeddings[:, w_list1_idx[:, 0], w_list1_idx[:, 1]],
                             embeddings[:, w_list2_idx[:, 0], w_list2_idx[:, 1]]).mean(dim=1)
            results[model] = cos_sim
            
    elif version == 'MEV':
        results = MEV_Anisotropy()

    return results


# Question 2 - main
def Anisotropy_function(version):
    results = anisotropy(version)
    draw_result(results, f'picture/Anisotropy_{version}.png')
    print(f'Finish Anisotropy of {version}!')
    return


def self_similarity():
    results = {}
    for model in MODELS:
        input_ids, embeddings = load_data(MODELS[model])
        sorted_ids, sorted_embeddings = arrange_data(input_ids, embeddings)
        
        _, counts = torch.unique(sorted_ids, return_counts=True)

        # release memory
        del input_ids
        del embeddings
        
        CosSim2D = CosineSimilarity(dim=2)
        CosSim3D = CosineSimilarity(dim=3)

        idx_start = 0
        idx_end = 0
        cos_sim = []
        for c in counts:
            idx_start = idx_end
            idx_end += c.item()  
                
            # if a word appears too many times (> 430 here), compute its CosineSimilarity 
            # layer by layer to avoid running out of memory.
            if c > 430:
                cos_sim_one_w = []
                for l in range(N_LAYER):
                    cos_sim_btn_em = CosSim2D(sorted_embeddings[l, idx_start:idx_end].unsqueeze(dim=1), 
                                              sorted_embeddings[l, idx_start:idx_end]).triu(diagonal=1)
                    cos_sim_one_w.append(cos_sim_btn_em.sum() / cos_sim_btn_em.count_nonzero())
                cos_sim_one_w = torch.stack(cos_sim_one_w).reshape(-1, 1)  # (13, 1)
            else:
                cos_sim_btn_em = CosSim3D(sorted_embeddings[:, idx_start:idx_end].unsqueeze(dim=1), 
                                          sorted_embeddings[:, idx_start:idx_end].unsqueeze(dim=2)).triu(diagonal=1)
                cos_sim_one_w = cos_sim_btn_em.sum(dim=(1, 2)) / cos_sim_btn_em.count_nonzero(dim=(1, 2))
                cos_sim_one_w = cos_sim_one_w.reshape(-1, 1)  # (13, 1)
            cos_sim.append(cos_sim_one_w)
            
            # release memory
            if c > 200:
                release_gpu_memory()

        cos_sim = torch.cat(cos_sim, dim=1).mean(dim=1)
        results[model] = cos_sim
        
    return results 


# Question 3 - main - SelfSimilarity
def SelfSimilarity_function():
    results = self_similarity()
    draw_result(results, 'picture/Self-similarity.png')
    print('Finish Self-similarity!')
    return     

    
def intrasentence_similarity():
    results = {}
    for model in MODELS:
        input_ids, embeddings = load_data(MODELS[model])
        
        # mask padding, [CLS] and [SEP]
        mask = (input_ids != 0) & (input_ids != 101) & (input_ids != 102)
        valid_words_idx = mask.nonzero()
        
        _, inversed_idx, counts = torch.unique(valid_words_idx[:, 0], return_inverse=True, return_counts=True)

        zero_tensor = torch.zeros(N_LAYER, N_DATA, N_FEATURE).to('cuda')
        s_embeddings = zero_tensor.index_add_(
            1, inversed_idx, embeddings[:, valid_words_idx[:, 0], valid_words_idx[:, 1]]).div_(counts.reshape(-1, 1))  # (13, 1000, 768)

        CosSim = CosineSimilarity(dim=2)
        zero_tensor = torch.zeros(N_LAYER, N_DATA).to('cuda')
        # cos sim of each word and its sentence, (13, # of all valid words)
        cos_sim_s_vs_w = CosSim(torch.repeat_interleave(s_embeddings, counts, dim=1), embeddings[:, valid_words_idx[:, 0], valid_words_idx[:, 1]])
        cos_sim = zero_tensor.index_add_(1, inversed_idx, cos_sim_s_vs_w).div_(counts).mean(dim=1)
        results[model] = cos_sim
        
    return results


# Question 3 - main - IntraSentenceSimilarity
def IntraSentenceSimilarity_function():
    results = intrasentence_similarity()
    draw_result(results, 'picture/Intra-sentence_similarity.png')
    print('Finish Intra-sentence similarity!')
    return    
    
    
# Question 4 - main - AnisotropyAdjustedSelfSimilarity
def AnisotropyAdjustedSelfSimilarity_function():
    anis = anisotropy(version='self-sim')
    release_gpu_memory()
    self_sim = self_similarity()
    # release_gpu_memory()
                
    results = {model: self_sim[model] - anis[model] for model in MODELS}
    draw_result(results, 'picture/Anisotropy-adjusted_self-similarity.png')
    print('Finish Anisotropy-adjusted self-similarity!')
    return 


# Question 4 - main - AnisotropyAdjustedIntraSentenceSimilarity
def AnisotropyAdjustedIntraSentenceSimilarity_function():
    anis = anisotropy(version='intra-sentence-sim')
    release_gpu_memory()
    intra_s_sim = intrasentence_similarity()
    # release_gpu_memory()
                
    results = {model: intra_s_sim[model] - anis[model] for model in MODELS}
    draw_result(results, 'picture/Anistropy-adjusted_intra-sentence_similarity.png')
    print('Finish Anistropy-adjusted intra-sentence similarity!')
    return 


### Bonus:
def MEV():
    results = {}
    for model in MODELS:
        input_ids, embeddings = load_data(MODELS[model], device='cpu')
        sorted_ids, sorted_embeddings = arrange_data(input_ids, embeddings)
        
        _, counts = torch.unique(sorted_ids, return_counts=True)
        
        idx_end = torch.tensor([n for n in accumulate(counts)])
        idx_start_end = torch.stack((idx_end-counts, idx_end), dim=1)  # start & end idx of each word
        
        results[model] = calculate_MEV(sorted_embeddings, idx_start_end)
        
    return results


### Bonus - 1: 
def MEV_Anisotropy():
    results = {}
    for model in MODELS:
        input_ids, embeddings = load_data(MODELS[model], device='cpu')
        sorted_ids, sorted_embeddings = arrange_data(input_ids, embeddings)
    
        _, counts = torch.unique(sorted_ids, return_counts=True)

        idx_end = torch.tensor([n for n in accumulate(counts)])
        idx_start_end = torch.stack((idx_end-counts, idx_end), dim=1)  # start & end idx of each word
        
        # sample 1000 words randomly
        random.seed(0)
        samples = random.sample(range(len(counts)), 1000)
        samples_idx = idx_start_end[samples]
        
        results[model] = calculate_MEV(sorted_embeddings, samples_idx)
        
    return results
    

### Bonus - 2:
def MaximumExplainableVariance_function():
    results = MEV()    
    draw_result(results, 'picture/Maximum_Explainable_Variance.png')
    print('Finish Maximum Explainable Variance!')
    return 


### Bonus - 3:
def AnisotropyAdjustedMEV_function():
    anis = anisotropy(version='MEV')
    release_gpu_memory()
    mev = MEV()
    
    results = {model: mev[model] - anis[model] for model in MODELS}
    draw_result(results, 'picture/Anistropy-adjusted_Maximum_Explainable_Variance.png')
    print('Finish Anistropy-adjusted Maximum Explainable Variance!')
    return 


if __name__ == '__main__':
    os.makedirs('picture', exist_ok=True)
    
    # Question 2
    Anisotropy_function(version='self-sim')
    Anisotropy_function(version='intra-sentence-sim')
    release_gpu_memory()
    
    # Question 3
    SelfSimilarity_function()
    release_gpu_memory()
    IntraSentenceSimilarity_function()

    # Question 4
    AnisotropyAdjustedSelfSimilarity_function()
    AnisotropyAdjustedIntraSentenceSimilarity_function()
    
    # Bonus
    Anisotropy_function(version='MEV')

    # Bonus
    MaximumExplainableVariance_function()

    # Bonus
    AnisotropyAdjustedMEV_function()
