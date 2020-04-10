# A simple neural network module for relational reasoning

## Basic Information

| 引用情報 |                                                                                                                        |
| -------- | ---------------------------------------------------------------------------------------------------------------------- |
| 筆者     | Adam Ssantoro, David Raposo, David G.T. Barret, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap |
| 所属     | Deepmind                                                                                                               |
| 会議     | NIPS                                                                                                                   |
| 年       | 2017                                                                                                                   |
| 引用数   | 591                                                                                                                    |
| リンク   | https://papers.nips.cc/paper/7082-a-simple-neural-network-module-for-relational-reasoning                              |

## どんなもの

Relational reasoning (関係推論)は通常のNeural Networksでは学習が難しい。筆者らが提案するRelation Networks(RNs)はrelational reasoningを解くことができるplug-and-playなモジュールである。筆者らはRNsを組み込んだネットワークでVisual QA(CLEVR)において人間を超える性能を達成した他、テキストベースのQA(bAbI)および動的な物理系に関する複雑な推論タスクを試している。また、Sort-of-CLEVRと呼ばれるデータセットを作成し、強力なCNNでも関係推論は行えないことを示した上でRNsを組み込むことで扱えるようになることを示した。

## 先行研究に比べてどこがすごい

関係推論は日常的に発生するタスクである。例えば、ミステリー小説を読む人は各証拠をグローバルなコンテキストの中でミステリーを解くための答えと関係付けなければならない。
Symbolic AIは本来的にrelationalであり、これらのrelationを論理などで表現し、関係を推論や代数的なアプローチで求めようとしたが、symbol grounding problemが解決できなかった。一方、統計的学習では生のデータから汎化する表現を作ることができる(symbol grounding problemを解決できそう)が、隠されている構造が複雑な関係に特徴付けられていながらデータがスパースであるときなどにうまくいかなくなる。

Relation Networksは関係推論に明示的にフォーカスしたアーキテクチャである一方で、同様の目的のアーキテクチャであるGraph Neural NetworksやGated Graph Sequence Neural Networks, Interaction Networksと比べてシンプルで、plug-and-playである。さらに、joint trainingを通して、RNは上流のCNNやLSTMなどにrelational reasoningの役に立つobject-likeな表現を生み出させることができる。

筆者らはCLEVRと呼ばれるVisual QAのデータセットで人間を超えるスコアを出した他、テキストベースのQAでも非常に良い成績を出している。

## 技術や手法のキモはどこ

Relation Networksはネットワークのアーキテクチャを以下の形に限定する。

<img src="https://latex.codecogs.com/gif.latex?\mathrm{RN}(O)&space;=&space;f_\phi\left(\sum_{i,j}g_\theta(o_i,o_j)\right)">

## どうやって有効だと検証した

## 議論はある

## 次に読むべき論文は
