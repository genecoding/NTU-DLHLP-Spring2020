# Homework 4
BERT
* [4-1]
* [4-2 & 4-3]

## Note
### 4-1
#### 4-1-1
* similarity_student.py (offered from TA) vs similarity.py (coded by myself)
  * Code in MEV() and MEV_Anisotropy() is ambiguous and seems incorrect, I tried my best guess... Also, the code uses word embedding to calculate MEV, 
    but [the paper](#Reference) uses occurrence matrix... anyway, done.
  * self_similarity()
    * I was running the code on the Colab, due to GPU memory constraint I had to do it that way. If there is sufficient GPU memory, it can be coded this way:
      ```
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
#### 4-1-2
### 4-2
### 4-3

## Reference
[How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings][p1], K Ethayarajh



[4-1]: https://docs.google.com/presentation/d/1WfZhcWykHiHoRdM26EUcyhAXY9inurIDE4tBoX1U2t0
[4-2 & 4-3]: https://docs.google.com/presentation/d/1IlNqFNknS1BvsDsuuUrYzPLfarakmMoZSJz-egvGbnw
[p1]: https://arxiv.org/abs/1909.00512
