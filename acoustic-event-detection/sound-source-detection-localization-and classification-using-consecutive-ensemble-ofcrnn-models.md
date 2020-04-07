# SOUND SOURCE DETECTION, LOCALIZATION AND CLASSIFICATION USING CONSECUTIVE ENSEMBLE OF CRNN MODELS

## Basic Information

| 引用情報 |                                                                                         |
| -------- | --------------------------------------------------------------------------------------- |
| 筆者     | Slawomir Kapka, Mateusz Lewandowski                                                     |
| 所属     | Samsung R&D Institute Poland                                                            |
| 会議     | Detection and Classification of Acoustic Scenes and Events 2019 (DCASE2019)             |
| 年       | 2019                                                                                    |
| 引用数   | -                                                                                       |
| リンク   | http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Kapka_26.pdf |

## どんなもの

DCASE2019 Challenge Task3 (Sound Event Localization and Detection)のトップsolutionのテクニカルレポート。
Sound Event Localization and Detectionは音声の空間的方向も考慮する場合に発生する複雑なタスク。

## 技術や手法のキモはどこ

SELD (Sound Event Localization and Detection)のタスクを、

1. 音源数の推定
2. 音がした方向の推定
3. 一つ目の音源の方向がわかっている時の二つ目の音源の方向の推定
4. マルチラベル推定

に分解した。これを4つの連続したCRNN SELDnetっぽいモデルで推定した。

## どうやって有効だと検証した

TAU Sound Events 2019 Ambisonicで評価。

## 議論はある

## 次に読むべき論文は
