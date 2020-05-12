# Meta-Learning: A Survey

## Basic Information

| 引用情報 |                                    |
| -------- | ---------------------------------- |
| 筆者     | Joaquin Vanschoren                 |
| 所属     | Eindhoven University of Technology |
| 会議     | -                                  |
| 年       | 2018                               |
| 引用数   | 77                                 |
| リンク   | https://arxiv.org/abs/1810.03548   |

## どんなもの

Meta-Learningのサーベイ論文。

### Meta Learning

我々は新しいタスクに直面した時に過去に学習した経験を元に、過去にうまくいった学習アプローチなどに注力して学習を行う。すなわち我々は*どのようにして学習をするのか*ということを学習している。meta-learningのchallengeはどのようにして過去の経験からシステマチックに、かつdata-drivenな方法で学習を行うか、という点にある。

第一に、我々は*meta-data*を収集する必要がある。このmeta-dataは過去に学習したタスクと学習したモデルに関する情報を含んでいる必要がある。これらの情報には、学習したアルゴリズムの構成 -

* hyperparameter
* pipeline構成 and/or ネットワークのアーキテクチャ
* モデルの評価結果
  * accuracy
  * training time
  * 学習されたモデルのパラメータ(学習済みモデルの重みなど)
* タスク自体の属性(*meta-features*)

などが含まれる。
次にこれらの*meta-data*を用いて学習を行う。

タスクが似ていれば似ているほど多くの*meta-data*を活用することができる。このタスクの類似度をどのように定義するか、というのが大きなチャレンジである。

## Learning from Model Evaluations



## 次に読むべき論文は
