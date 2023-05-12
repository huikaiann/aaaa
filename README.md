# aaaa

.. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j


Note: Source function taken from PyTorch Geometric and modified such that
    embeddings are first concatenated and then reduced to out_channel size as
    per the original GraphSAGE paper.
    源函数取自PyTorch Geometric，并进行了修改，使得嵌入首先被连接，
    然后根据原始GraphSAGE论文减少到out_channel大小

    GraphSAGE来自于
    The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        每个输入样本的大小。元组对应于源维度和目标维度的大小。

        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    PyTorch Geometric citation:
    @inproceedings{Fey/Lenssen/2019,
      title={Fast Graph Representation Learning with {PyTorch Geometric}},
      author={Fey, Matthias and Lenssen, Jan E.},
      booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
      year={2019},
    }
