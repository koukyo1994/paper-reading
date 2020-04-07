# LEARNING TO MAKE ANALOGIES BY CONTRASTING ABSTRACT RELATIONAL STRUCTURE

## Basic Information

| 引用情報 |                                                                          |
| -------- | ------------------------------------------------------------------------ |
| 筆者     | Felix Hill, Adam Santoro, David G. T. Barrett, Ari Morcos, Tim Lillicrap |
| 所属     | Deepmind                                                                 |
| 会議     | ICLR                                                                     |
| 年       | 2019                                                                     |
| 引用数   | 9                                                                        |
| リンク   | https://arxiv.org/abs/1902.00120                                         |

## どんなもの

比喩を用いることは様々なドメインに対して柔軟に適用されうる関係構造の習得が必要なため、機械にとっては非常に難しい。比喩表現は、細かい部分では様々な差がある二つのものの間に柔軟な関連づけ、あるいはマッピングを見出す行為である。この論文はNeural Networksに類推を行わせる方法論を検討した物である。
これによれば、類推を行わせる上で最も重要なのは、複雑なアーキテクチャではなくデータの選び方とそれがいかにモデルに与えられるか、ということである。筆者らは、最も頑健に類推を学習させることができたのは、比喩的な関係構造を比較するように与えた時であった。筆者らはこの訓練法を*learning analogies by contrasting abstract relational structre*(LABC)と呼び単純なアーキテクチャのニューラルネットでも今まで見たことのないsource-target domain mappingができ、時には全く新しいtarget domainに対しても適用できることを示した。

### 比喩(analogy)と類似性(similarity)の違い

Structure Mapping Theoryによれば、二つのdomainの間に多くの共有の属性がある場合にはsimilarであると呼ばれる一方、二つの間に共通する属性は少ないが、共通の関係が多くある場合にはanalogousであると呼ばれる。

一方High-Level Perception(HLP)理論によれば、analogyは知覚と推論が緊密に相互作用する関数である。例えば海と音のアナロジーを考えるとき、私たちは特定の知覚的特徴を表現し、他を切り捨てる。

## 先行研究に比べてどこがすごい

* 先行研究と違い、表現の学習とドメイン間マッピングをjointで行う。これにより、認知・表現・ドメイン間の対応づけの間にある関係を学習できる可能性がある。
* アーキテクチャに明示的に類推を行うのに利するような機構を組み込むのではなく、モデルの学習をいかに行うかという点に類推を行うための鍵を仕込んだ(LABC)。

## 技術や手法のキモはどこ

### データについて

人間に対しても使われる知覚テストのような物を利用した。各sceneは3枚の*panel*(異なる画像)の列からなる*source sequence*と*target sequence*(2枚の*panel*からなる)、そして4枚の候補panelからなる。source sequenceの中には`R={XOR, OR, AND, Progression}`からなる関係rが表現されており、それとtarget sequenceおよび候補panelを見てtarget sequenceを完成させるのに必要な候補panelを選び直す必要がある。

visual analogy taskにおいては*domain*は`line type`, `line color`, `shape type`, `shape color`, `shape size`, `shape quality`, `shape position`からなる。あるdomainのpanelでは、scene中の属性は10の値のどれかをとる。

## どうやって有効だと検証した

Meauring abstract reasoning in neural networksで提案したPGMデータセットを用いている。

## 議論はある

## 次に読むべき論文は
