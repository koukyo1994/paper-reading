# PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition

## Basic Information

| 引用情報 |                                                                                 |
| -------- | ------------------------------------------------------------------------------- |
| 筆者     | Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbeley |
| 所属     | IEEE                                                                            |
| 会議     | -                                                                               |
| 年       | 2020                                                                            |
| 引用数   | -                                                                               |
| リンク   | https://arxiv.org/abs/1912.10211                                                |

## どんなもの

Audio pattern recognition用の大規模pretrained model PANNsの提案。Wavegramと呼ばれるwaveformから学習された特徴とmel spetrogramを入力とするモデルをAudioSetで学習し、AudioSetのタグ付けタスクでSoTAを達成したほか、6つの他のタスクへの転移学習でもSoTAを達成した。

PANNsが適用された転移学習先はacousti scene classification, general audio tagging, music classification, speech emotion classificationなどである。

## 先行研究に比べてどこがすごい

これまではSound pattern recognitionの問題において大規模なデータセットで学習されたpretrainedモデルが適用されることは限定的であった。近年公開されたAudioSetは5000時間以上、527のあとクラスを収録した大規模なデータセットであり、生の音ではなく音クリップからCNNを用いて抽出した特徴が公開されている点が特徴的である。
この論文は今までの研究では限定的であった、音パターン認識問題において大規模な事前学習モデルを用いた点が特徴的である。

## 技術や手法のキモはどこ

## どうやって有効だと検証した

## 議論はある

## 次に読むべき論文は
