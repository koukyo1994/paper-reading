# Non-local Neural Networks

## Basic Information

| 引用情報 |                                                                                                          |
| -------- | -------------------------------------------------------------------------------------------------------- |
| 筆者     | Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He                                                  |
| 所属     | Carnegie Mellon University, Facebook AI Research                                                         |
| 会議     | CVPR                                                                                                     |
| 年       | 2018                                                                                                     |
| 引用数   | 1117                                                                                                     |
| リンク   | http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf |

## どんなもの

non-local演算を行うことで長距離依存性を扱うニューラルネットワークの構成ブロックの提案論文。コンピュータビジョンの世界でnon-local means methodとして知られる手法を参考にしており、ある位置における出力が全ての位置の特徴の重み付き和になる。様々なCVのアーキテクチャに接続可能でビデオ分類ではKineticsとCharadesデータセットで既存のSoTAを更新する性能を出した。また、画像認識においては、object detection/segmentation, pose estimationで既存のモデル(Mask R-CNN)の性能を向上させることが確認された。

検証は画像と動画で行われているが、実際には画像、系列データ、動画に適用できるだろうと述べられている。non-local operationsの強みとしては以下の３つが挙げられている。

1. CNNやRNNと異なり、non-local operationsでは直接長距離依存性をモデリングできる
2. 効率がよく、層が少なくてもいい性能が出せる
3. 入力が可変長であるため、様々な演算と結合が可能である

### 課題感

長距離依存性を捉える必要性はDeep Learningにおいてあちこちで出現する。系列データ(音声、言語など)ではRNNの利用(今だとattentionか？)が長距離依存性を捉えるために用いられる一方で画像データではこれまでは大きな局所受容野をCNNで実現することでモデリングされてきた。

CNNやRNNは局所近傍に対して処理が適用されるため、長距離依存性を扱うためには繰り返し演算が適用される必要があるがこれは３つの点で問題がある。

1. 繰り返しの計算は非効率である。
2. 最適化が難しくなる
3. multi-hop dependency、すなわち離れた場所の間でメッセージが行き来しなければならない場合にモデリングが難しくなる

## 先行研究に比べてどこがすごい

### 関連研究

Non-local meansはフィルタリングアルゴリズムの一つで画像中の全てのピクセルの重みつき平均を計算する。これによりパッチの見た目の類似性に基づいて離れたピクセルがフィルタされた出力に寄与することができる。これはデノイズのアルゴリズムとして知られるBM3Dのアイデアとしても採用されている。

長距離依存性を扱うのは、CRFや多層のFeed-Foward model、Self Attention, Interaction-NetworksやRelation Networksなどによりモデル化されている。Attention/interaction/relation networksとの違いはnon-localityとのこと(本当か？？？？？？)

## 技術や手法のキモはどこ

non-local operationは以下のようにして定義される。

![non-local operation](https://latex.codecogs.com/gif.latex?y_i&space;=&space;\frac{1}{c(x)}\sum_{\forall&space;j}f(x_i,&space;x_j)g(x_j))

ここで`i`, `j`は出力中での場所を表しており、![y](https://latex.codecogs.com/gif.latex?y)は![x](https://latex.codecogs.com/gif.latex?x)と同じサイズの出力である。関数`f`はスカラーを計算し、`g`は地点`j`の入力信号の表現を計算する。これらは`C(x)`により標準化をされる。

(これrelation networksでは？)

`f`や`g`の構成の仕方はいくつかあるがそこまで重要ではない。論文中では`g`は普通の線形変換![g](https://latex.codecogs.com/gif.latex?g(x_j)&space;=&space;W_g&space;x_j)である。

## どうやって有効だと検証した

## 議論はある

## 次に読むべき論文は
