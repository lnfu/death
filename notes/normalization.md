# Normalization

negative binomial distribution = gamma-Poisson

負二項分佈包含兩個參數：
1. μ_ij：平均值
2. α_i：離散度（每個基因一個，但在不同樣本間相同）

而平均值 μ_ij 又由兩部分組成：
1. q_ij：樣本中該基因的 cDNA 片段濃度（真實的生物學資訊）
2. s_ij：size factor（用來校正偏差）

μ_ij = s_ij × q_ij

簡化假設每個樣本使用單一常數 s_j
=> 校正樣本間的定序深度（sequencing depth）差異
=> 同一樣本中的所有基因使用相同 size factor

如何估計 size factor？

DESeq/DESeq2 使用 median-of-ratios method

補充：cqn 或 EDASeq 方法或許更好估計出 size factor？

