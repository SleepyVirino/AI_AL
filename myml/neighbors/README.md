# K近邻算法

## 基本概念

直接比较样本与所有训练样本的相似度，然后根据与它最相似的$k$个训练样本，进行分类或者回归任务。

由于该算法没有训练过程，只有在预测时参与对训练样本的处理和计算，被称为**懒惰学习**，与之相对的被称为**急切学习**。

## 预测算法

给定$l$个训练样本$(\mathbf{x_i},y_i)$，其中$\mathbf{x_i}$为特征向量，$y_i$为标签值或回归值，待预测的样本为$\mathbf{x}$，预测算法如下：

1. 在训练集中找到距离$\mathbf{x}$最近的$k$个样本，样本集合为$N$。

2. 对于分类与回归任务：

   - **分类**：统计集合$N$中每一个样本的个数$C_i,i=1,...,c$，预测结果为
     $$
     \underset{i\in [c]}{argmaxC_i}
     $$

   - **回归**：预测结果为$N$中$y$的均值或加权平均
     $$
     \overline{y}=(\sum_{i=1}^{k}y_i)/k
     $$

     $$
     \hat{y}=(\sum_{i=1}^{k}w_iy_i)/k
     $$

     其中$w_i$人工设定，比如可设置为与距离成反比

## 距离定义

距离函数$d(\mathbf{x_i},\mathbf{x_j})$需要满足四个条件：

- 三角不等式：$d(\mathbf{x_i},\mathbf{x_k})+d(\mathbf{x_k},\mathbf{x_j})\geq d(\mathbf{x_i},\mathbf{x_j})$
- 非负性：$d(\mathbf{x_i},\mathbf{x_j})\geq 0$
- 对称性：$d(\mathbf{x_i},\mathbf{x_j})=d(\mathbf{x_j},\mathbf{x_i})$
- 区分性：$d(\mathbf{x_i},\mathbf{x_j})=0 \rightarrow \mathbf{x_i}=\mathbf{x_j}$

### 常用距离定义

**欧氏距离（L2距离）**
$$
d(\mathbf{x},\mathbf{y})=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$
欧氏距离只是将特征向量看作空间中的点，并未考虑这些样本特征向量的概率分布规律

**Mahalanobis距离**
$$
d(\mathbf{x},\mathbf{y})=\sqrt{(\mathbf{x-y})^T\mathbf{S}(\mathbf{x-y})}
$$
要求矩阵$S$必须是半正定的，当$S=I$时，Mahalanobis距离退化为欧氏距离。

