# li-stephen-hmm-replication
This is a python version of Li-Stephen-hmm implementation

## Genotype imputation
Define hidden stage as ![equation](https://latex.codecogs.com/png.latex?\dpi{150}&space;\bg_white&space;\pmb{x}), which has ![equation](https://latex.codecogs.com/png.latex?\dpi{150}&space;\bg_white&space;K) different reference haplotype stages; define ![equation](https://latex.codecogs.com/png.latex?\dpi{150}&space;\bg_white&space;\pmb{o}_{1,S}) as the observational sequence. Then the genotype imputation is the argmax of the posterior probability:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}&space;\bg_white&space;p(x_j=k&space;|&space;h_{k,\leq&space;M})&space;\propto&space;P(x_j=k,&space;h_{k,\leq&space;j})P(h_{k,[j+1,M]}|x_j=k))

**Forward path**:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}&space;\bg_white&space;\alpha_{j}(x)\equiv&space;P(x_{j}=x,&space;h_{k+1,\leq&space;j})&space;=&space;\sum_{k'\in&space;K}P(h_{k+1,&space;j}|&space;x_{j}=x)P(x_{j}=x|x_{j-1}=k')P(x_{j-1}=k',h_{k+1,\leq&space;j})&space;=&space;\gamma_{j}(x)(p_{j-1}\alpha_{j-1}(k)&space;+&space;(1-p_{j-1})\frac{1}{k}\sum_{k'}\alpha_{j-1}(k'))

where ![equation](https://latex.codecogs.com/png.latex?\dpi{150}&space;\bg_white&space;\gamma_{j+1}(x)&space;=&space;Pr(h_{k+1,j+1}&space;|&space;X_{j+1}=x,h_1,...,h_k))

**Backward path**:

\begin{align*}
\beta_{j}(x) \equiv P(h_{k+1,[j,M]}|X_{j-1}=x) & = \sum_{k' \in K}P(h_{k+1,[j+1,M]}|X_j=k') P(h_{k+1,j}|x_{j}=k')P(x_{j}=k'|X_{j-1}=x) \\
& = \sum_{k' \in K} \beta_{j+1}(k')\gamma_{j}(k')P(X_j=k' | X_{j-1}=x) \\
&= p_{j-1}\beta_{j+1}(k)\gamma_j(k) + (1-p_{j-1})\frac{1}{K}\sum_{k'}\beta_{j+1}(k')\gamma_{j}(k')
\end{align*}

where ![equation](https://latex.codecogs.com/png.latex?\dpi{150}&space;\bg_white&space;p_j&space;=&space;\exp(-\rho_jd_j/K)) and ![equation](https://latex.codecogs.com/png.latex?\dpi{150}&space;\bg_white&space;\gamma_j(x)&space;=&space;Pr(h_{k+1,j+1}&space;|&space;X_{j+1}=x,h_1,...,h_k))

**Log forward path**:

![equation](https://latex.codecogs.com/svg.image?\begin{align*}\log(\alpha_j(k))&=\log\gamma_j(k)&plus;\log\left(p_{j-1}\alpha_{j-1}(k)&plus;(1-p_{j-1})\frac{1}{K}\sum_{k'}\alpha_{j-1}(k')\right)\\&=\log\gamma_i(k)&plus;\log\left(\exp(-\rho_jd_j/K)\exp(\log(\alpha_{j-1}(k)))&plus;\frac{1}{K}\sum_{k'}\exp(\log\alpha_{j-1}(k'))-\exp(-\rho_jd_j/K)\sum_{k'}\exp(\log\alpha_{j-1}(k'))/K\right)\\&=\log\gamma_i(k)&plus;\log\left(\exp(-\rho_jd_j/K&plus;\log\alpha_{j-1}(k))&plus;\sum_{k'}\exp(\log\alpha_{j-1}(k'))/K-\sum_{k'}\exp(\log\alpha_j(k')-\rho_jd_j/K)/K\right)\end{align*})

**Posterior calculation**
![equation](https://latex.codecogs.com/svg.image?\begin{align*}p(h_{j}=1|h_{1,M})&=\sum_{k\in&space;K}p(h_j=1,x_j=k|h_{1,M},h_{train})\\&=\sum_{k\in&space;K}p(h_j=1|x_j=k,h_{1,M},h_{train})p(x_j=k|h_{1,M})\\&=\sum_{k\in&space;K}\mathbf{1}[h_{x,j}=1]\left(K/(K&plus;\tilde{\theta})&plus;(1/2)\tilde{\theta}/(K&plus;\tilde{\theta})\right)p(x_j=k|h_{1,M})\end{align*})

