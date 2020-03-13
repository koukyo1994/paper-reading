# Recent Advances in Neural Program Synthesis
## Basic Information
### 引用情報
Kant, Neel. "Recent advances in neural program synthesis." arXiv preprint arXiv:1802.02353 (2018).
###  リンク
https://arxiv.org/abs/1802.02353
### Citation
7
## どんな分野?
### Program Synthesisとは
"The task of developing an algorithm that meets a specification or a set of constraints".
制約条件を満たすようなアルゴリズムが出力となるタスク。**制約条件には**、導出速度やcomplexityなどが含まれる他、**ほぼ必ずアルゴリズムに対する(入力, 出力)ペアが含まれる。**
### 応用
プログラミングの自動化ができるはず。コンピュータにreasoningやlogic, automationなどを行わせることを目標とした分野である。
### 類似の分野
Black Boxの存在を仮定する。そのBlack Boxに入力と出力のペアを与えた時に、
* 入力から出力までのロジックが辿れるような出力をBlack Boxが返す場合、これはProgram Synthesisである。Interpretabilityが高い。
* 入力を与えると、出力の予測値をBlack Boxが返す場合、これはProgram Inductionである。Deep Learningなどはこっち。
### Deep Learningの発展と関連して
Deep Learningが得意とする分野とは大きく異なり難しい。Deep Learningが得意なSpeech Recognition, Computer Vision, NLPなどは入力空間が連続であり、その空間にentities with meaningが分布している。
このように**continuous, high-dimensional, distributed**表現を扱う物を**connectionist AI**という。
Program Synthesisはこれとは異なり、continuous domain spaceを持たない。多様体上に埋め込むのもの難しい。算術/論理演算子を用いる。
## 過去にはどんな手法があった?
古くは1960年代に始まる**Symbolic AI**(⇆Connectionist AI)の研究に始まる。Continuousな表現を避けていた。CFGに従うDomain Specific Language(DSL)を用いていた。
学習が難しいという問題があり、ルールベースでプログラムを構成し、解空間を狭くして探索するというアプローチが取られた。現在最も使われるツールは**Satisfiability Modulo Theories(SMT)** solversと呼ばれる。
Symbolic AIは学習プロセスが限られていたとは言え、汎化性能と証明可能性に大きなアドバンテージがある。現代のハードウェアを用いれば非常に高速に動作するのも確か。
ただし、Connectionist AIとSymbolic AIはどちらが正しいという物ではなく互いに補い合う関係にある。したがって**Hybrid System**がいいだろう。Symbolをprogram state spaceに埋め込む。
## 最新手法はどんな感じ?
## 次に読むべき論文は?
