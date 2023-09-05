![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/toba/framework.png)

<h1 align="center">
  ToBA: Topological Augmentation for Class-Imbalanced Node Classification
</h1>

<h3 align="center">
Links: [<a href="https://arxiv.org/abs/2308.14181">arXiv</a>] [<a href="https://arxiv.org/pdf/2308.14181.pdf">PDF</a>] [<a href="https://github.com/ZhiningLiu1998/ToBA/blob/main/toba.py#L170">Implementation</a>]
</h3>

**ToBA** (Topological Balanced Augmentation) is a **lightweight, plug-and-play** graph data augmentation technique for class-imbalanced node classification tasks. It aims to mitigate the **class-imbalance bias** introduced by **ambivalent and distant message-passing** by dynamically identifying and rectifying nodes that are exposed to such issues. Our **ToBA** implementation features:

- &#x1F34E; **Plug-and-play**: model-agnostic augmentation that directly integrates into the training loop.
- &#x1F34E; **Effectiveness**: boosting classification performance, while reducing predictive bias.
- &#x1F34E; **Versatility**: work with various GNN architectures and imbalance-handling techniques.
- &#x1F34E; **Lightweight**: light computational overhead, no additional hyperparameters.
- &#x1F34E; **Ease-of-use**: unified, concise, and extensible API design.

Intergrating [`TopoBalanceAugmenter`](https://github.com/ZhiningLiu1998/ToBA/blob/main/toba.py#L170) (**ToBA**) into your training loop with <5 lines of code:

```python
from toba import TopoBalanceAugmenter

augmenter = TopoBalanceAugmenter().init_with_data(data)

for epoch in range(epochs):
    # augment the graph
    x, edge_index, _ = augmenter.augment(model, x, edge_index)
    y, train_mask = augmenter.adapt_labels_and_train_mask(y, train_mask)
    # original training step
    model.update(x, y, edge_index, train_mask)
```

### Table of Contents
- [Background](#background)
- [Examples Scripts](#examples-scripts)
  - [Command line](#command-line)
  - [Jupyter Notebook](#jupyter-notebook)
- [API reference](#api-reference)
  - [TopoBalanceAugmenter](#topobalanceaugmenter)
  - [NodeClassificationTrainer](#nodeclassificationtrainer)
- [Emprical Results](#emprical-results)
  - [Experimental Setup](#experimental-setup)
  - [On the effectiveness and versatility of TOBA](#on-the-effectiveness-and-versatility-of-toba)
  - [On the robustness of TOBA](#on-the-robustness-of-toba)
  - [On mitigating AMP and DMP](#on-mitigating-amp-and-dmp)
- [Citing us](#citing-us)
- [References](#references)

## Background

Class imbalance is prevalent in real-world node classification tasks and often biases graph learning models toward majority classes. Most existing studies root from a node-centric perspective and aim to address the class imbalance in training data by node/class-wise reweighting or resampling. In this paper, we approach the source of the class-imbalance bias from an under-explored topology-centric perspective. 

Our investigation reveals that beyond the inherently skewed training class distribution, the graph topology also plays an important role in the formation of predictive bias: we identify two fundamental challenges, namely ambivalent and distant message-passing, that can exacerbate the bias by aggravating majority-class over-generalization and minority-class misclassification. 

![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/toba/motivation.png)


In light of these findings, we devise a lightweight topological augmentation method TOBA to dynamically rectify the nodes influenced by ambivalent/distant message-passing during graph learning, so as to mitigate the class-imbalance bias. We highlight that TOBA is a model-agnostic, efficient, and versatile solution that can be seamlessly combined with and further boost other imbalance-handling techniques. Systematic experiments validate the superior performance of TOBA in both promoting imbalanced node classification and mitigating the prediction bias between different classes.

For more details on the algorithm design and analysis, please refer to the paper: https://arxiv.org/abs/2308.14181.

## Examples Scripts

### Command line

[`train.py`](https://github.com/ZhiningLiu1998/ToBA/blob/main/train.py) provides a simple way to test ToBA under different settings: datasets, imbalance types, imbalance ratios, GNN architectures, etc. For example, to test ToBA's effectiveness on the Cora dataset with a 10:1 step imbalance ratio using the GCN architecture, simply run:
```bash
python train.py --dataset cora --imb_type step --imb_ratio 10 --gnn_arch GCN --toba_mode all
```

Output:
```
================= Dataset [Cora] - StepIR [10] - ToBA [dummy] =================
Best Epoch:   97 | train/val/test | ACC: 100.0/67.20/67.50 | BACC: 100.0/61.93/60.55 | MACRO-F1: 100.0/59.65/59.29 | upd/aug time: 4.67/0.00ms | node/edge ratio: 100.00/100.00% 
Best Epoch:   67 | train/val/test | ACC: 100.0/65.20/65.00 | BACC: 100.0/60.04/57.70 | MACRO-F1: 100.0/57.21/55.09 | upd/aug time: 3.36/0.00ms | node/edge ratio: 100.00/100.00% 
Best Epoch:  131 | train/val/test | ACC: 100.0/66.80/67.90 | BACC: 100.0/63.78/61.71 | MACRO-F1: 100.0/62.26/60.08 | upd/aug time: 3.37/0.00ms | node/edge ratio: 100.00/100.00% 
Best Epoch:   60 | train/val/test | ACC: 100.0/66.40/66.30 | BACC: 100.0/61.60/60.74 | MACRO-F1: 100.0/58.04/59.09 | upd/aug time: 3.34/0.00ms | node/edge ratio: 100.00/100.00% 
Best Epoch:  151 | train/val/test | ACC: 100.0/63.40/63.70 | BACC: 100.0/58.00/55.99 | MACRO-F1: 100.0/53.57/51.88 | upd/aug time: 3.19/0.00ms | node/edge ratio: 100.00/100.00% 
Avg Test Performance (5 runs):  | ACC: 66.08 ± 0.70 | BACC: 59.34 ± 0.96 | MACRO-F1: 57.09 ± 1.40

================== Dataset [Cora] - StepIR [10] - ToBA [pred] ==================
Best Epoch:   95 | train/val/test | ACC: 100.0/64.80/63.70 | BACC: 100.0/63.14/60.69 | MACRO-F1: 100.0/60.22/58.30 | upd/aug time: 3.48/3.58ms | node/edge ratio: 100.26/103.05% 
Best Epoch:  157 | train/val/test | ACC: 100.0/71.80/69.70 | BACC: 100.0/71.59/68.44 | MACRO-F1: 100.0/69.45/66.74 | upd/aug time: 3.36/3.64ms | node/edge ratio: 100.26/103.19% 
Best Epoch:  177 | train/val/test | ACC: 100.0/73.40/73.20 | BACC: 100.0/73.27/71.69 | MACRO-F1: 100.0/71.31/70.53 | upd/aug time: 3.34/3.64ms | node/edge ratio: 100.26/102.89% 
Best Epoch:  340 | train/val/test | ACC: 100.0/70.20/73.00 | BACC: 100.0/65.76/67.88 | MACRO-F1: 100.0/64.42/67.45 | upd/aug time: 3.41/3.84ms | node/edge ratio: 100.26/103.13% 
Best Epoch:   90 | train/val/test | ACC: 100.0/66.60/67.30 | BACC: 100.0/61.18/59.96 | MACRO-F1: 100.0/58.85/58.07 | upd/aug time: 3.19/3.65ms | node/edge ratio: 100.26/103.23% 
Avg Test Performance (5 runs):  | ACC: 69.38 ± 1.60 | BACC: 65.73 ± 2.06 | MACRO-F1: 64.22 ± 2.28

================== Dataset [Cora] - StepIR [10] - ToBA [topo] ==================
Best Epoch:   72 | train/val/test | ACC: 100.0/72.00/72.20 | BACC: 100.0/69.65/68.93 | MACRO-F1: 100.0/66.88/67.10 | upd/aug time: 3.12/4.10ms | node/edge ratio: 100.26/101.43% 
Best Epoch:  263 | train/val/test | ACC: 100.0/72.80/71.70 | BACC: 100.0/72.59/69.01 | MACRO-F1: 100.0/72.05/68.70 | upd/aug time: 3.51/4.10ms | node/edge ratio: 100.26/101.75% 
Best Epoch:  186 | train/val/test | ACC: 100.0/74.00/73.70 | BACC: 100.0/74.37/73.10 | MACRO-F1: 100.0/71.61/71.04 | upd/aug time: 3.36/4.15ms | node/edge ratio: 100.26/101.56% 
Best Epoch:   71 | train/val/test | ACC: 100.0/72.40/72.10 | BACC: 100.0/69.50/67.75 | MACRO-F1: 100.0/68.11/66.80 | upd/aug time: 3.31/4.12ms | node/edge ratio: 100.26/101.55% 
Best Epoch:   77 | train/val/test | ACC: 100.0/76.20/77.60 | BACC: 100.0/78.03/77.92 | MACRO-F1: 100.0/75.06/76.42 | upd/aug time: 3.34/4.10ms | node/edge ratio: 100.26/101.58% 
Avg Test Performance (5 runs):  | ACC: 73.46 ± 0.97 | BACC: 71.34 ± 1.68 | MACRO-F1: 70.01 ± 1.58
```

Full argument list and descriptions are as follows:

```
--gpu_id | int, default=0
    Specify which GPU to use for training. Set to -1 to use the CPU.

--seed | int, default=42
    Random seed for reproducibility in training.

--n_runs | int, default=5
    The number of independent runs for training.

--debug | bool, default=False
    Enable debug mode if set to True.

--dataset | str, default="cora"
    Name of the dataset to use for training.
    Supports "cora," "citeseer," "pubmed," "cs", "physics".

--imb_type | str, default="step", choices=["step", "natural"]
    Type of imbalance to handle in the dataset. Choose from "step" or "natural".

--imb_ratio | int, default=10
    Imbalance ratio for handling imbalanced datasets.

--gnn_arch | str, default="GCN", choices=["GCN", "GAT", "SAGE"]
    Graph neural network architecture to use. Choose from "GCN," "GAT," or "SAGE."

--n_layer | int, default=3
    The number of layers in the GNN architecture.

--hid_dim | int, default=256
    Hidden dimension size for the GNN layers.

--lr | float, default=0.01
    Initial learning rate for training.

--weight_decay | float, default=5e-4
    Weight decay for regularization during training.

--epochs | int, default=2000
    The number of training epochs.

--early_stop | int, default=200
    Patience for early stopping during training.

--tqdm | bool, default=False
    Enable a tqdm progress bar during training if set to True.

--toba_mode | str, default="all", choices=["dummy", "pred", "topo", "all"]
    Mode of the ToBA. Choose from "dummy," "pred," "topo," or "all."
    if "dummy," ToBA is disabled.
    if "pred," ToBA is enabled with only prediction-based augmentation.
    if "topo," ToBA is enabled with only topology-based augmentation.
    if "all," will run all modes and report the result for comparison.
```

### Jupyter Notebook

We also provide an example jupyter notebook [train_example.ipynb](https://github.com/ZhiningLiu1998/ToBA/blob/main/train_example.ipynb) with experimental results on:
- Datasets:        ['cora', 'citeseer', 'pubmed']
- ToBA modes:      ['dummy', 'pred', 'topo']
- Imbalance types: 
  - 'step': [10, 20]
  - 'natural': [50, 100]

## API reference

### TopoBalanceAugmenter

https://github.com/ZhiningLiu1998/ToBA/blob/main/toba.py#L170

Main class that implements the ToBA augmentation algorithm, inheriting from [`BaseGraphAugmenter`](https://github.com/ZhiningLiu1998/ToBA/blob/main/toba.py#L11). Implements 3 core steps of ToBA: 
- (1) node risk estimation
- (2) candidate class selection
- (3) virtual topology augmentation.

```python
class TopoBalanceAugmenter(BaseGraphAugmenter):
    """
    Topological Balanced Augmentation (ToBA) for graph data.

    Parameters:
    - mode: str, optional (default: "pred")
        The augmentation mode. Must be one of ["dummy", "pred", "topo"].
    - random_state: int or None, optional (default: None)
        Random seed for reproducibility.
    """
```

Core methods:
- `init_with_data(data)`: initialize the augmenter with graph data.
  - Parameters: 
    - `data` : PyG data object
  - Return: 
    - `self` : TopoBalanceAugmenter
- `augment(model, x, edge_index)`: perform topology-aware graph augmentation.
  - Parameters: 
    - `model` : torch.nn.Module, node classification model
    - `x` : torch.Tensor, node feature matrix
    - `edge_index` : torch.Tensor, sparse edge index
  - Return: 
    - `x_aug` : torch.Tensor, augmented node feature matrix
    - `edge_index_aug`: torch.Tensor, augmented sparse edge index
    - `info` : dict, augmentation info
- `adapt_labels_and_train_mask(y, train_mask)`: adapt labels and training mask after augmentation.
  - Parameters: 
    - `y` : torch.Tensor, node label vector
    - `train_mask` : torch.Tensor, training mask
  - Return: 
    - `new_y` : torch.Tensor, adapted node label vector
    - `new_train_mask` : torch.Tensor, adapted training mask

### NodeClassificationTrainer

https://github.com/ZhiningLiu1998/ToBA/blob/main/trainer.py#L14

Trainer class for node classification tasks, centralizing the training workflow: 
- (1) model preparation and selection
- (2) performance evaluation
- (3) data augmentation
- (4) verbose logging.

```python
class NodeClassificationTrainer:
    """
    A trainer class for node classification with Graph Augmenter.

    Parameters:
    -----------
    - model: torch.nn.Module
        The node classification model.
    - data: pyg.data.Data
        PyTorch Geometric data object containing graph data.
    - device: str or torch.device
        Device to use for computations (e.g., 'cuda' or 'cpu').
    - augmenter: BaseGraphAugmenter, optional
        Graph augmentation strategy.
    - learning_rate: float, optional
        Learning rate for optimization.
    - weight_decay: float, optional
        Weight decay (L2 penalty) for optimization.
    - train_epoch: int, optional
        Number of training epochs.
    - early_stop_patience: int, optional
        Number of epochs with no improvement to trigger early stopping.
    - eval_freq: int, optional
        Frequency of evaluation during training.
    - eval_metrics: dict, optional
        Dictionary of evaluation metrics and associated functions.
    - verbose_freq: int, optional
        Frequency of verbose logging.
    - verbose_config: dict, optional
        Configuration for verbose logging.
    - save_model_dir: str, optional
        Directory to save model checkpoints.
    - save_model_name: str, optional
        Name of the saved model checkpoint.
    - enable_tqdm: bool, optional
        Whether to enable tqdm progress bar.
    - random_state: int, optional
        Seed for random number generator.
    """
```

Core methods:
- `train`: train the node classification model and perform evaluation.
  - Parameters:
    - `train_epoch`: int, optional. Number of training epochs.
    - `eval_freq`: int, optional. Frequency of evaluation during training.
    - `verbose_freq`: int, optional. Frequency of verbose logging.
  - Return:
    - `model`: torch.nn.Module, trained node classification model.
- `print_best_results`: print the evaluation results of the best model.

## Emprical Results

### Experimental Setup

To fully validate **ToBA**'s performance and compatibility with existing IGL techniques and GNN backbones, we test 6 baseline methods with 5 popular GNN backbone architectures in our experiments, and apply ToBA with them under all possible combinations:

- **Datasets**: Cora, Citeseer, Pubmed, CS, Physics
- **Imbalance-handling techniques**: 
  - Reweighting [1]
  - ReNode [2]
  - Oversample [3]
  - SMOTE [4]
  - GraphSMOTE [5]
  - GraphENS [6]
- **GNN backbones**:
  - GCN [7]
  - GAT [8]
  - SAGE [9]
  - APPNP [10]
  - GPRGNN [11]
- **Imbalance types & ratios**: 
  - **Step imbalance**: 10:1, 20:1
  - **Natural imbalance**: 50:1, 100:1

For more details on the experimental setup, please refer to our paper: https://arxiv.org/abs/2308.14181.

### On the effectiveness and versatility of TOBA

We first report the detailed empirical results of applying **ToBA** with 6 IGL baselines and 5 GNN backbones on 3 imbalanced graphs (Cora, CiteSeer, and PubMed) with IR=10 in Table 1. In all 3 (datasets) x 5 (backbones) x 7 (baselines) x 2 (name variants) x 3 (metrics) = **630 setting combinations**, it achieves significant and consistent performance improvements on the basis of other IGL techniques, which also yields new state-of-the-art performance. In addition to the superior performance in boosting classification, **ToBA** also greatly reduces the model predictive bias.

![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/toba/table1.png)


### On the robustness of TOBA

We now test **ToBA**'s robustness to varying types of extreme class-imbalance. In this experiment, we extend Table `main` and consider a more challenging scenario with IR = 20. In addition, we consider the natural (long-tail) class imbalance that is commonly observed in real-world graphs with IR of 50 and 100. Datasets from (*CS, Physics*) are also included to test **ToBA**'s performance on large-scale tasks. Results show that **ToBA** consistently demonstrates superior performance in boosting classification and reducing predictive bias.

![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/toba/table8.png)


### On mitigating AMP and DMP

We further design experiments to verify whether **ToBA** can effectively handle the topological challenges identified in this paper, i.e., ambivalent and distant message-passing. Specifically, we investigate whether **ToBA** can improve the prediction accuracy of minority class nodes that are highly influenced by ambivalent/distant message-passing, i.e., high local heterophilic ratios/long distance to supervision signals. Results are shown in the figure below (5 independent runs with GCN classifier, IR=10). As can be observed, **ToBA** effectively alleviates the negative impact of AMP and DMP and helps node classifiers to achieve better performance in minority classes.

![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/toba/mitigatebias.png)

Please refer to our paper: https://arxiv.org/abs/2308.14181 for more details, including

- full experimental results and discussions
- complexity analysis
- scalability results
- reproducibility details
- further discussions on speedup of ToBA
- limitations and future works


## Citing us

If you find this repository useful in your research, we would appreciate citation to our work:
```bibtex
@article{liu2023topological,
  title={Topological Augmentation for Class-Imbalanced Node Classification},
  author={Liu, Zhining and Zeng, Zhichen and Qiu, Ruizhong and Yoo, Hyunsik and Zhou, David and Xu, Zhe and Zhu, Yada and Weldemariam, Kommy and He, Jingrui and Tong, Hanghang},
  journal={arXiv preprint arXiv:2308.14181},
  year={2023}
}
```

## References

| #    | Reference                                                                                                                                                                                                                                  |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [1]  | Nathalie Japkowicz and Shaju Stephen. The class imbalance problem: A systematic study. Intelligent data analysis, 6(5):429–449, 2002.                                                                                                      |
| [2]  | Deli Chen, Yankai Lin, Guangxiang Zhao, Xuancheng Ren, Peng Li, Jie Zhou, and Xu Sun. Topology-imbalance learning for semi-supervised node classification. Advances in Neural Information Processing Systems, 34:29885–29897, 2021.        |
| [3]  | Nathalie Japkowicz and Shaju Stephen. The class imbalance problem: A systematic study. Intelligent data analysis, 6(5):429–449, 2002.                                                                                                      |
| [4]  | Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16:321–357, 2002.                                               |
| [5]  | Tianxiang Zhao, Xiang Zhang, and Suhang Wang. Graphsmote: Imbalanced node classification on graphs with graph neural networks. In Proceedings of the 14th ACM international conference on web search and data mining, pages 833–841, 2021. |
| [6]  | Joonhyung Park, Jaeyun Song, and Eunho Yang. Graphens: Neighbor-aware ego network synthesis for class-imbalanced node classification. In International Conference on Learning Representations, 2022.                                       |
| [7]  | Max Welling and Thomas N Kipf. Semi-supervised classification with graph convolutional networks. In J. International Conference on Learning Representations (ICLR 2017), 2016.                                                             |
| [8]  | Petar Veliˇckovi ́c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks. In International Conference on Learning Representations, 2018.                                            |
| [9]  | Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. Advances in neural information processing systems, 30, 2017.                                                                             |
| [10] | Johannes Gasteiger, Aleksandar Bojchevski, and Stephan Günnemann. Predict then propagate: Graph neural networks meet personalized pagerank. arXiv preprint arXiv:1810.05997, 2018.                                                         |
| [11] | Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. Adaptive universal generalized pagerank graph neural network. arXiv preprint arXiv:2006.07988, 2020.                                                                               |
