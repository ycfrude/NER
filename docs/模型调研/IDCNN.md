# IDCNN
原始论文为[ITERATED DILATED CONVOLUTIONAL NEURAL NETWORKS FOR WORD SEGMENTATION](https://sceweb.sce.uhcl.edu/xiaokun/doc/Publication/2021/NNW2021_HHe.pdf) 。

## 论文内容
论文的基调是尽管Bi-LSTM类的序列化编码器在大规模的数据集上取得了非常不错的效果，但是rnn类模型的串行结构没有办法完全发挥GPU的性能。

传统的cnn类模型的卷积核计算在GPU上可以高并发实现，但受限于**感受野**的问题，在单个token上只能获得附近w个token的位置信息，如果要扩大信息的范围，就要不断加深cnn的层数，l层的cnn的感受野为l(w-1)-1,随着文本长度的增长，代价会线性增加，加上深层网络的更加敏感，训练难度也会提高。

在cnn中，把每个filter的dilation变大，就可以增加cnn的感受野，但是仍然会面临cnn层数上升，导致的过拟合的问题。所以idcnn就对卷积核进行**重用**，但改变卷积核的dilation，减少参数量，减少过拟合。(注:cnn的过拟合还可以通过残差连接,normalization或者pooling缓解，本文并没有使用)。

以上的为核心思想。本文除此之外还提到了 自由基[radical embedding](https://aclanthology.org/P15-2098.pdf) ，暂时还不了解这个东西是做什么的。

速度上idcnn+crf大约是bilstm+crf的两倍，实验结果上idcnn+crf的效果略优于bilstm+crf，总体上持平。
除此之外，idcnn的性能总体上与idcnn+crf持平，bilstm的性能则差于bilstm+crf很多，作者认为idcnn由此并不依赖于crf。
## 个人感受
这篇文章行文上比较粗糙，有几处勘误。除此之外，作者在试验说明这一块做的并不详尽：
- 参数方面，模型的max_length与cnn层数还有dilation的选择都没有说明清楚。
- 也没有对超参数的调优方案作出说明，超参数的选择范围的指导性意见也无从谈起。
- 作者的bilstm试验结果并不来自于引用结果，bilstm的超参数也没有提供。
- 作者的试验结果是单次试验得出，还是多次试验取平均或者取最佳也没有说明。

总的来说，这篇文章的核心思想较为简单直接，试验部分并不严谨，也不充分，调优技巧用的也不多，所以没有得到广泛的关注。美团搜索当前的实体识别模型采用的是bert蒸馏的idcnn，总体性能与bert接近。这个点说明了这个模型的潜力。后续可以继续探索。

