# 贝叶斯

```html
Author:	杜耀达
Email:	d837791568yd@gmai.com
```



## 贝叶斯决策

​		假设有$N$种可能的类别标记，即$Y=\{c_1,c_2,...,c_N\}$，$\lambda_{ij}$是将一个真实标记为$c_j$的样本误分类为$c_i$的样本所产生的损失。基于后验概率$P(c_j|\mathbf{x})$可获得将样本分类为$c_i$所产生的期望损失，即在样本$\mathbf{x}$上的“条件风险”。
$$
R(c_i|\mathbf{x})= \sum_{j=1}^{N}\lambda_{ij} P(c_j|\mathbf{x})
$$
我们的任务是寻找判定准则$h:X \rightarrow Y$以最小化总体风险
$$
R(h)=E_x[R(h(\mathbf{x})|\mathbf{x}]
$$
显然，对每个样本$\mathbf{x}$，若$h$能最小化条件风险$R(h(\mathbf{x})|\mathbf{x})$，则总体风险$R(h)$也将被最小化。这就产生了**贝叶斯判定准则**：为最小化总体风险，只需要在每个样本上选择那个使条件风险$R(x|\mathbf{x})$最小的类别标记，即
$$
h^*(x) = \underset{c\in Y}{arg min}R(c|\mathbf{x})
$$
此时，$h^*$称为**贝叶斯最优分类器**，与之相应的总体风险$R(h^*)$称为贝叶斯风险。$1-R(h^*)$反映了分类器所能达到的最好性能，即通过机器学习能产生的模型精度的理论上限。

​		具体来说，若目标是最小化分类错误率，则误判损失$\lambda_{ij}$可写为
$$
\lambda_{ij} =
\begin{cases}
  0, & \text{if } i=j ; \\
  1, & \text{otherwise, }
\end{cases}
$$
此时条件风险
$$
R(c|\mathbf{x})=1-P(c|\mathbf{x})
$$
于是最小化分类错误率的贝叶斯最优分类器为
$$
h^*(\mathbf{x})=\underset{c\in Y}{arg max}P(c|\mathbf{x})
$$
即对每个样本$x$，选择能够使后验概率$P(c|\mathbf{x})$最大的类别标记

对于后验概率$P(c|\mathbf{x})$的估计，有两种策略

- **判别式模型**：对给定$x$，直接对$P(c|\mathbf{x})$进行建模来预测$c$
- **生成式模型**：对给定$x$，对联合概率分布$P(\mathbf{x},c)$进行建模，然后再获得$P(c|\mathbf{x})$

对于生成式模型，考虑
$$
P(c|\mathbf{x})=\frac{P(\mathbf{x},c)}{P(\mathbf{x})}=\frac{P(c)P(\mathbf{x}|c)}{P(\mathbf{x})}
$$
$P(c)$为类概率，这里又称为先验概率，可从样本中直接用频率估计得出

$P(\mathbf{x}|c)$为类条件概率，或称为似然，这里不能使用频率来估计（样本空间太大！）

## 极大似然估计

估计类条件概率的一种常用策略：假定其具有某种确定的概率分布形式，再基于训练样本对概率分布的参数进行估计，如：

假设$P(\mathbf{x}|c)$服从某种分布或（具有某种具体形式），并且该分布（形式）被参数向量$\mathbf{\theta_c}$唯一确定，则只需估计参数$\mathbf{\theta_c}$即可

事实上，概率模型的训练过程就是参数估计的过程，对于参数估计，两种方法：

- 频率主义学派：参数虽然未知，但是确客观存在，可以优化似然函数来确定参数值
- 贝叶斯学派：参数本身也具有某种分布，可以假设参数服从一个先验分布，然后再计算参数的后验分布

**极大似然法（MLE）**是根据数据采样来估计概率分布参数的经典办法

$D_c$表示训练集$D$中第$c$类样本组成的集合，假设这些样本独立同分布，则参数$\mathbf{\theta_c}$对于数据集$D_c$的似然是
$$
P(D_c|\mathbf{\theta_c})=\prod_{\mathbf{x}\in D_c}P(\mathbf{x}|\mathbf{\theta_c})
$$
对$\mathbf{\theta_c}$进行极大似然估计，使用对数似然
$$
LL(\mathbf{\theta_c})=logP(D_c|\mathbf{\theta_c})=\sum_{\mathbf{x}\in D_c}log(P(\mathbf{x}|\mathbf{\theta_c}))
$$
此时参数$\mathbf{\theta_c}$的极大似然估计$\mathbf{\hat{\theta_c}}$为
$$
\mathbf{\hat{\theta_c}}=\underset{\mathbf{\theta_c}}{argmax}LL(\mathbf{\theta_c})
$$
可使用两种办法来求解该问题：

- 直接求解：如特征向量属性值连续时，假设概率密度函数$p(\mathbf{x}|c)\sim N(\mathbf{\mu_c},\mathbf{\sigma_c^2})$，则此时可直接解出$\mathbf{\hat{\theta_c}}=\{\mathbf{\hat{\mu_c}},\mathbf{\hat{\sigma_c^2}}\}$
  $$
  \mathbf{\hat{\mu_c}}=\frac{1}{|D_c|} \sum_{\mathbf{x}\in D_c}\mathbf{x}\\
  \mathbf{\hat{\sigma_c^2}}=\frac{1}{|D_c|} \sum_{\mathbf{x}\in D_c}\mathbf(x-\hat{\mu_c})(x-\hat{\mu_c)}^T
  $$

- 通过最优化算法进行求解

## 朴素贝叶斯分类器

**属性条件独立性假设**：对已知类别，假设所有属性相互独立，即
$$
P(\mathbf{x}|c) = \prod_{i=1}^{d}P(x_i|c)
$$
则朴素贝叶斯分类器的表达式为：
$$
h_{nb}=\underset{c\in Y}{argmax}P(c|\mathbf{x})=P(c)\prod_{i=1}^{d}P(x_i|c)
$$
令$D_c$表示训练集$D$中第c类样本组成的集合，类先验概率
$$
P(c)=\frac{|D_c|}{|D|}
$$
对于$P(x_i|c)$，分两种情况：

- 特征向量$x_i$为离散值：记$D_{c,x_i}$表示$D_c$中在第$i$个属性取值为$x_i$的样本集合，则

​		
$$
P(x_i|c)=\frac{|D_{c,x_i}|}{|D_c|}
$$

- 特征向量$x_i$为连续值：则可假定$p(x_i|c)\sim N(\mu_{c,i},\sigma_{c,i}^2)$，其中$\mu_{c,i}$和$\sigma_{c,i}^2$分别是第$c$类样本在第$i$个属性上取值的均值和方差，则：

$$
p(x_i|c)=\frac{1}{\sqrt{2\pi\sigma_{c,i}^2}}exp(-\frac{{(x_i-\mu_{c,i})}^2}{2\sigma_{c,i}^2})
$$

**拉普拉斯平滑**：修正$P(c)$和$P(x_i|c)$，这里是因为某种类别或属性值在训练集中并未出现，则会导致这两个式子值为0，则会导致整体概率为0，显然这样是不合理的（会导致某些样本$x_i$无论无何都不会分到类别$c$中），修正之后为：
$$
\hat{P}(c)=\frac{|D_c|+1}{|D|+N}\\
\hat{P}(x_i|c)=\frac{|D_{c,x_i}|+1}{|D_c|+N_i}
$$
其中$N$为训练集$D$中可能出现的类别数，$N_i$表示第$i$个属性可能的取值个数

## 正态贝叶斯分类器

特征向量服从多维正态分布，即：
$$
P(\mathbf{x}|c)=\frac{1}{{(2\pi)}^{\frac{1}{2}}|\mathbf{\Sigma_c}|}exp(-\frac{1}{2}(\mathbf{x}-\mathbf{\mu_c})^T\Sigma^{-1}(\mathbf{x}-\mathbf{\mu_c}))
$$
其中$\mathbf{\mu_c}$为类别为$c$时的均值向量，$\mathbf{\Sigma_c}$为此时的协方差矩阵

并且由极大似然估计可得样本中的均值向量与协方差矩阵即为上述所需值

则分类器$h_g$为：
$$
h_g(x)=\underset{c\in Y}{argmax}P(c|\mathbf{x})=\underset{c\in Y}{argmax}\frac{P(c)P(\mathbf{x}|c)}{P(\mathbf{x})}=\underset{c\in Y}{argmax}P(c)P(\mathbf{x}|c)
$$
取对数似然：
$$
h_g(x)=\underset{c\in Y}{argmin}(\ln(|\mathbf{\Sigma_c}|)+(\mathbf{x}-\mathbf{\mu_c})^T\Sigma^{-1}(\mathbf{x}-\mathbf{\mu_c}))-\ln(P(c))
$$


当数据中各个类别概率近似相等时，即$P(c)$是相等的，那么上述式子只需求前半部分即可
