# Universal Sound Separation

## Basic Information

| 引用情報 |                                                                                                            |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| 筆者     | Ilya Kavalerov, Scott Wisdom, Hakan Erdogan, Brian Patton, Kevin Wilson, Jonathan Le Roux, John R. Hershey |
| 所属     | Google Research, UMD, Mitsubishi Electric Research Laboratories                                            |
| 会議     | IEEE Workshop on Applications of Signal Processing ti Audio and Acoustics(WASPAA)                          |
| 年       | 2019                                                                                                       |
| 引用数   | 12                                                                                                         |
| リンク   | https://arxiv.org/pdf/1905.03330.pdf                                                                       |

## どんなもの

音声に限らず様々な種類の混合に対して音源分離を行う方法論を検討した論文。新たなデータセットの提案と、mask-based separationのアーキテクチャの探索を行っている。探索を行ったアーキテクチャとしてはConvolutional-LSTMや、近年成功を収めているdilated convolutionを利用した、時間領域における強調ネットワークConvTasNetなどを扱っている。

また、信号変換のためのframewise analysis-synthesis basis(?)に関しても研究を行い、音源分離のパフォーマンスをあげるような変更を提案している。framewise analysis-synthesis basisに関してはSTFTとConvTasNetなどで用いられるlearnable basisの両方を検証しており窓幅の効果を検証した。
特にSTFTでは、窓幅が(25-50ms)と大きい時はspeech / non-speech分離に向いている一方で窓幅が小さい(2.5ms)では様々な音の分離に向いていることを発見した。Universal Sound Separation (様々な種類の音の混合に対する音源分離)ではSTFTの方がよかった。

## 前提

モノラルの音源分離はめっちゃ難しいが、近年はDeep Learningにより目覚ましい発展を遂げている。一方で特定の音種を仮定しない音源分離はまだめっちゃむずいとされている。

ConvTasNetがめっちゃ強いらしい。これを前提にしているのでこっちを先に読んだ方がいいかも。

## 先行研究に比べてどこがすごい

過去の音源分離の問題設定では少なくとも一つの音源は音声であるという仮定があった。複数話者の分離では未知の話者に関する分離もできる方向で発展が進み、最近提案されたConvTasNetでは、time-dilated convolutional network(TDCN)とlearnableなtime-domain analysis, synthesis transformの組み合わせにより過去のSTFT + LSTMベースのネットワークに比べて大きな進歩を遂げた。
一方で、音声以外の音の分離に関してはどうなのか今までよくわかっていなかった。

この論文では、

1. Universal Sound Separationという問題設定で初めてしっかりと 検証を行った
2. ConvTasNetをspeech/non-speech separationでもuniversal sound separationの問題設定でも検証した
3. 様々なmasking network アーキテクチャとanalysis-synthesis変換の組み合わせを検証し、窓幅に関して最適な解を発見した
4. 新規なアーキテクチャ変更を提案した

## 技術や手法のキモはどこ

mask-basedな分離システムに関して検討をしている。その際に二組のネットワーク構成と二組のanalysis-synthesis basisの組み合わせを検討している。

## どうやって有効だと検証した

## 議論はある

## 次に読むべき論文は
