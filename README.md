# HSTU 生成式推荐模型复现（基于Keras）

从零复现并实现 Meta 最新提出的 HSTU（Hierarchical Sequential Transduction Unit）推荐模型，完整覆盖数据预处理、模型构建、训练评估与推理优化全流程。

深入理解并实现了 HSTU 的核心创新机制：相对位置偏置与对数分桶时间偏置（捕捉序列顺序与时间衰减）、去 Softmax 的注意力（SiLU 激活保留多兴趣强度）、门控结构替代 FFN（降低计算量）、输出层与物品嵌入权重绑定（减少参数量）。

数据层面采用留一法（Leave-One-Out）评估协议，使用稳定排序确保实验可复现性；模拟动作类型区分正负反馈，训练时仅从正向样本中学习，使模型聚焦真实用户兴趣。

实现 M‑FALCON 推理优化，通过缓存用户隐状态将候选物品打分复杂度从 O(batch × num_items × d_model) 降至 O(batch × num_candidates × d_model)，为大规模候选集的高效排序奠定基础。

在 MovieLens 数据集上完成端到端训练与评估，验证了生成式推荐在序列建模中的可行性，并系统记录了从论文到代码的复现经验与挑战。

## 技术栈：TensorFlow / Keras, Pandas, NumPy, Scikit-learn, TF Data Pipeline

# HSTU Generative Recommendation Model: From Paper to Code

Implemented from scratch Meta's HSTU (Hierarchical Sequential Transduction Unit) recommendation model, covering the complete pipeline of data preprocessing, model construction, training, evaluation, and inference optimization.

Deeply understood and implemented core innovations of HSTU: relative position bias and logarithmic time bucketing (capturing sequential order and temporal decay), removal of Softmax in attention (SiLU activation for multi-interest modeling), gating mechanism replacing FFN (reducing computation), and weight tying between output layer and item embeddings (parameter efficiency).

Adopted Leave-One-Out evaluation protocol with stable sorting to ensure reproducibility; simulated action types to distinguish positive/negative feedback and trained exclusively on positive samples to focus on genuine user interests.

Implemented M‑FALCON inference optimization, reducing candidate scoring complexity from O(batch × num_items × d_model) to O(batch × num_candidates × d_model) by caching user-side hidden states, enabling efficient ranking over large-scale candidate sets.

Completed end-to-end training and evaluation on the MovieLens dataset, validating the feasibility of generative recommendation in sequential modeling, and systematically documented the experience and challenges of translating a research paper into executable code.

## Tech Stack: TensorFlow / Keras, Pandas, NumPy, Scikit-learn, TF Data Pipeline

