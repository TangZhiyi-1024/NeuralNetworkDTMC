### model

```
sequence_model.h5
```

w= 64, d=2, loss=mse , bs=8, optimizer= adam, earlystop+rlr

```
sequence_model_1.h5
```

w= 64, d=2, loss=mse , bs=8, optimizer= adam, earlystop+rlr

```
sequence_model_2.h5
```

w= 32, d=1, loss=mse  , bs=32, optimizer= adam, earlystop+rlr

```
sequence_model_3.h5
```

w= 32, d=1, loss=huber, bs=32, optimizer= adam, earlystop+rlr

```
sequence_model_4.h5
```

w= 32, d=2, loss=huber, bs=32, optimizer= adam, earlystop+rlr

```
sequence_model_5.h5
```

w= 64, d=3, loss=mse , bs=8, optimizer= adam, earlystop+rlr





### Result

== Traditional DTMC ==
N_valid            : 130
NLL (mean)        : 4.640166   | Uniform: 4.852030
Brier (mean)      : 0.986038
Top-k 命中        : @1=0.138, @3=0.192, @5=0.223
期望预测 RMSE/MAE : 1054.908 / 843.716
容差准确率(|err|≤200) : 0.162
平均熵(锐度)      : 4.816
ECE(top1)         : 0.112

== NN_base+Gauss DTMC ==
N_valid            : 130
NLL (mean)        : 8.757535   | Uniform: 4.852030
Brier (mean)      : 1.007739
Top-k 命中        : @1=0.169, @3=0.254, @5=0.300
期望预测 RMSE/MAE : 990.682 / 659.030
容差准确率(|err|≤200) : 0.323
平均熵(锐度)      : 3.095
ECE(top1)         : 0.126

== NN_v1+Gauss DTMC ==
N_valid            : 130
NLL (mean)        : 8.837996   | Uniform: 4.852030
Brier (mean)      : 1.013530
Top-k 命中        : @1=0.177, @3=0.254, @5=0.315
期望预测 RMSE/MAE : 1000.847 / 664.733
容差准确率(|err|≤200) : 0.331
平均熵(锐度)      : 3.070
ECE(top1)         : 0.147

== NN_v2+Gauss DTMC ==
N_valid            : 130
NLL (mean)        : 8.833834   | Uniform: 4.852030
Brier (mean)      : 1.013615
Top-k 命中        : @1=0.177, @3=0.246, @5=0.315
期望预测 RMSE/MAE : 1000.999 / 664.535
容差准确率(|err|≤200) : 0.331
平均熵(锐度)      : 3.070
ECE(top1)         : 0.161

== NN_v3+Gauss DTMC ==
N_valid            : 130
NLL (mean)        : 8.836720   | Uniform: 4.852030
Brier (mean)      : 1.014264
Top-k 命中        : @1=0.177, @3=0.246, @5=0.315
期望预测 RMSE/MAE : 1001.657 / 665.245
容差准确率(|err|≤200) : 0.331
平均熵(锐度)      : 3.069
ECE(top1)         : 0.148

== NN_v4+Gauss DTMC ==
N_valid            : 130
NLL (mean)        : 8.835511   | Uniform: 4.852030
Brier (mean)      : 1.013971
Top-k 命中        : @1=0.177, @3=0.246, @5=0.315
期望预测 RMSE/MAE : 1001.064 / 664.999
容差准确率(|err|≤200) : 0.331
平均熵(锐度)      : 3.069
ECE(top1)         : 0.148

== NN_v5+Gauss DTMC ==
N_valid            : 130
NLL (mean)        : 8.836443   | Uniform: 4.852030
Brier (mean)      : 1.013836
Top-k 命中        : @1=0.177, @3=0.246, @5=0.315
期望预测 RMSE/MAE : 1000.906 / 664.903
容差准确率(|err|≤200) : 0.331
平均熵(锐度)      : 3.071
ECE(top1)         : 0.148

==Summary (sorted by TolAcc desc) ==
NN_v1+Gauss DTMC     | TolAcc=0.3308 | MAE=664.7 | Top1=0.177 | NLL=8.838
NN_v2+Gauss DTMC     | TolAcc=0.3308 | MAE=664.5 | Top1=0.177 | NLL=8.834
NN_v3+Gauss DTMC     | TolAcc=0.3308 | MAE=665.2 | Top1=0.177 | NLL=8.837
NN_v4+Gauss DTMC     | TolAcc=0.3308 | MAE=665.0 | Top1=0.177 | NLL=8.836
NN_v5+Gauss DTMC     | TolAcc=0.3308 | MAE=664.9 | Top1=0.177 | NLL=8.836
NN_base+Gauss DTMC   | TolAcc=0.3231 | MAE=659.0 | Top1=0.169 | NLL=8.758
Traditional DTMC     | TolAcc=0.1615 | MAE=843.7 | Top1=0.138 | NLL=4.640



### Small Grid:

=== 结果排名（按 验证集原始尺度 MAE 从小到大） ===
 1. w= 32, d=1, loss=mse  , bs=32 | Val MAE(orig)=105.4081
 2. w= 64, d=1, loss=mse  , bs=32 | Val MAE(orig)=105.4480
 3. w= 32, d=1, loss=mse  , bs=64 | Val MAE(orig)=105.4590
 4. w=128, d=1, loss=huber, bs=32 | Val MAE(orig)=105.5162
 5. w= 64, d=1, loss=huber, bs=64 | Val MAE(orig)=105.6054
 6. w= 32, d=3, loss=huber, bs=64 | Val MAE(orig)=105.6462
 7. w= 64, d=1, loss=huber, bs=32 | Val MAE(orig)=105.7045
 8. w= 32, d=2, loss=mse  , bs=32 | Val MAE(orig)=105.8107
 9. w= 64, d=2, loss=huber, bs=64 | Val MAE(orig)=105.8125
10. w=128, d=3, loss=huber, bs=64 | Val MAE(orig)=105.9817

=== 最佳配置（固定 lr） ===
Val MAE(orig)=105.4081 | w=32, d=1, loss=mse, bs=32
已保存：best_grid_model_fixedlr.h5, best_x_scaler.pkl, best_y_scaler.pkl
[Baseline] random (original): 1321.6181