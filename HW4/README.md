# Homework 4
BERT
* [4-1]
* [4-2 & 4-3]

## Note
### 4-1
#### 4-1-1 (Natural Language Inference Task & Contextualization Issue)
* similarity_student.py (offered from TA) and similarity.py (coded by myself)
  * Code in `MEV()` and `MEV_Anisotropy()` in similarity_student.py is ambiguous and seems incorrect, I tried my best guess... Also, the code uses word embeddings to       calculate MEV, but [the paper](#Reference) uses occurrence matrix... anyway, done.
  * `self_similarity()` in similarity.py
    * I ran the code on the Colab, due to GPU memory constraint I cound only do it that way. If there is sufficient GPU memory, 
      it can be coded in a more elegant way:
      ```python
      def self_similarity():
          results = {}
          for model in MODELS:
              input_ids, embeddings = load_data(MODELS[model])
              sorted_ids, sorted_embeddings = arrange_data(input_ids, embeddings)

              _, counts = torch.unique(sorted_ids, return_counts=True)

              idx_end = torch.tensor([n for n in accumulate(counts)])
              idx_start_end = torch.stack((idx_end-counts, idx_end), dim=1)  # start & end idx of each word

              CosSim = CosineSimilarity(dim=3)
              cos_sim = []
              for idx in idx_start_end:
                  cos_sim_btn_em = CosSim(sorted_embeddings[:, idx[0]:idx[1]].unsqueeze(dim=1), 
                                          sorted_embeddings[:, idx[0]:idx[1]].unsqueeze(dim=2)).triu(diagonal=1)
                  cos_sim_one_w = cos_sim_btn_em.sum(dim=(1, 2)) / cos_sim_btn_em.count_nonzero(dim=(1, 2))
                  cos_sim_one_w = cos_sim_one_w.reshape(-1, 1)  # (13, 1)
                  cos_sim.append(cos_sim_one_w)
              cos_sim = torch.cat(cos_sim, dim=1).mean(dim=1)
              results[model] = cos_sim

          return results
      ```
#### 4-1-2 (Chinese Word Segmentation)
* Although TA asks us to put the post-processing of sequence label inside the model (in modeling_bert.py), I don't think it's a good idea, since it will affect training. It's better to put the post-processing outside the model (e.g. example.py). I actually tried the both, the result shows putting the post-processing inside the model deteriorates the performance (when the epoch becomes larger, the performance becomes worse).
  | epoch |  f1   | precision | recall |  
  |-------|-------|-----------|--------|  
  |2      |0.9605 |0.9631     |0.9580  |
  |4      |0.9417 |0.9420     |0.9414  |
  |6      |0.9029 |0.9053     |0.9005  |
* In fact, the model can learn valid (not definitely correct) tag sequences by itself without post-processing, and it's surely a better way to do.

### 4-2
* Bonus task: rank loss
  * $L_{dist}^{rank} = \sum_{i,j>i} [1 - sign(d_i-d_j)(\hat d_i-\hat d_j)]^+$ (d: depth)
    * A weird loss function... rank here is ambiguous, it looks like neither rank loss (hinge loss) in machine learning nor rank in a rank tree... anyway, done.

## Reference
* https://github.com/huggingface/transformers
* [How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings][p1], K Ethayarajh



[4-1]: https://docs.google.com/presentation/d/1WfZhcWykHiHoRdM26EUcyhAXY9inurIDE4tBoX1U2t0
[4-2 & 4-3]: https://docs.google.com/presentation/d/1IlNqFNknS1BvsDsuuUrYzPLfarakmMoZSJz-egvGbnw
[p1]: https://arxiv.org/abs/1909.00512
