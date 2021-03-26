# kaggle-RANZCR
<img width="947" alt="スクリーンショット 2021-02-23 18 48 36" src="https://user-images.githubusercontent.com/70050531/108826575-cbbde300-7607-11eb-9edd-35ac39b4e8cd.png">
Kaggleの RANZCR CLiP - Catheter and Line Position Challenge コンペのリポジトリです。<br>
nbというディレクトリに、今回使用したNotebookをおいてあります。<br>
ただし、下の方針にもある通り、今回はKaggle上でほぼ完結させているため、gitは使用していません。ですので、nbの中に全てのversionのNotebookがあるわけではないです。ver名がないものに関しては最新のverになります。

## 最終結果<br>
<img width="956" alt="スクリーンショット 2021-03-21 10 54 02" src="https://user-images.githubusercontent.com/70050531/111891151-f9921e00-8a33-11eb-8652-8c71b456b375.png">
- Public: 0.96864<br>
- Private: 0.97142<br>
- rank: 91/1547 (Top6%)<br> 
- 初メダルゲット！！！！！<br>

## 方針
- 基本的にはKaggle上で実験を行い、quotaが切れたらローカルのGPUで実験を行う。
- Version名は常にVersion〇〇に統一して、変更などはKaggle日記(このリポジトリのREADMD.md)に書き込む。<br> 

## Paper<br>
- 参考にした論文の一覧。決して全てを理解してるわけではない。<br>

| No | Name | Detail | Date | link |
| ----- | ----- | ----- | ----- | ----- |
| 01 | Focal Loss for Dense Object Detection | Cross Entropy Loss において、比較的うまく分類できているものの損失を小さく抑えることにより、不均衡データでうまく学習ができる。(もともとはRCNを対象に開発された。) | 7 Feb 2018 | [link](https://arxiv.org/pdf/1708.02002.pdf) | 
| 02 | Dual Attention Network for Scene Segmentation | 画像の各位置、チャンネル方向の２つに対してSelf Attentionを適用する手法。(もともとはSegmentationのための手法。) | 21 Apr 2019 | [link](https://arxiv.org/pdf/1809.02983.pdf) | 
| 03 | Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification | 従来のAUC最適化用の損失関数(AUC square loss)の代わりにAUC margin lossを用いることにより、大規模データでも安定してAUC最適化を行えるようにした手法。 | 6 Dec 2020 | [link](https://arxiv.org/pdf/2012.03173.pdf) | 
| 04 | Data Augmentation Revisited: Rethinking the Distribution Gap between Clean and Augmented Data | augmentationを使ってモデルを学習させた後、augmentationを外してfine tuning することにより精度を上げる手法。 | 21 Nov 2019 | [link](https://arxiv.org/pdf/1909.09148.pdf) | <br>

- その他に参考にしたサイトなど。<br>
  - 評価指標がAUCであるとき、アンサンブルの際に各モデルの予測をn(>1)乗してWeight Averageを取ることで精度が上がるという話([link](https://medium.com/data-design/reaching-the-depths-of-power-geometric-ensembling-when-targeting-the-auc-metric-2f356ea3250e))。ただし、lateサブしたらスコアが悪化した。<br>
  - roc starという、Deep AUC とはまた違うAUC用の損失関数([githubのlink](https://github.com/iridiumblue/roc-star))。こちらも実際に試したら精度が悪化した。<br>

## Basics<br>
### Overview(Deepl)<br>
深刻な合併症は、患者のラインやチューブの位置を間違えた結果として発生する可能性があります。医師や看護師は、患者を管理する際にプロトコルに従っていることを確認するために、救命器具を配置するためのチェックリストを頻繁に使用しています。しかし、これらの手順には時間がかかることがあり、特に病院の収容人数が多いストレスの多い状況では、人為的なミスが発生しやすい。<br> 

入院患者は入院中にカテーテルやラインを挿入されることがありますが、位置を誤ると重篤な合併症を引き起こす可能性があります。経鼻胃管の気道内への位置ずれは、症例の最大3％で報告されており、これらの症例の最大40％が合併症を示しています[1-3]。手術室外で挿管された成人患者の気道チューブの位置異常は、最大25％の症例で報告されています[4,5]。合併症の可能性は、手技者の経験レベルと専門性の両方に直接関係している。位置がずれたチューブを早期に発見することは、危険な合併症（死に至ることもある）を防ぐ鍵であり、何百万人ものCOVID-19患者がこれらのチューブやラインを必要としている現在ではなおさらである。<br> 

ラインとチューブの位置を確認するためのゴールドスタンダードは胸部X線写真である。しかし、医師や放射線技師は、ラインやチューブが最適な位置にあるかどうかを確認するために、これらの胸部X線写真を手動でチェックしなければならない。これはヒューマンエラーの可能性を残すだけでなく、放射線技師は他のスキャンの報告に忙しくなるため、遅延が生じることもよくあります。ディープラーニングアルゴリズムは、カテーテルやラインの位置が間違っていることを自動的に検出することができるかもしれない。警告が出れば、臨床医は命に関わる合併症を避けるために、カテーテルの位置を変更したり、除去したりすることができる。<br> 

Royal Australian and New Zealand College of Radiologists（RANZCR）は、オーストラリア、ニュージーランド、シンガポールの臨床放射線技師と放射線腫瘍医のための非営利の専門組織である。RANZCRは、不適切な位置に置かれたチューブやラインを予防可能なものとして認識している世界の多くの医療機関（NHSを含む）の一つである。RANZCRは、そのようなエラーがキャッチされる安全システムの設計を支援しています。<br> 

このコンペティションでは、胸部レントゲン上のカテーテルやラインの存在と位置を検出します。機械学習を使用して、40,000枚の画像上でモデルをトレーニングしてテストし、不適切な位置にあるチューブを分類します。<br> 

データセットには、ラベル付けとの整合性を確保するために、一連の定義でラベル付けを行いました。正常カテゴリには、適切に配置され、再配置を必要としない線が含まれます。境界線のカテゴリには，理想的には多少の再配置を必要とするが，ほとんどの場合，現在の位置でも十分に機能するであろう線が含まれる．異常カテゴリーには、直ちに再配置を必要とするラインが含まれる。<br> 

成功すれば、臨床医の命を救うことができるかもしれない。COVID-19の症例が急増し続ける中、位置がずれたカテーテルやラインを早期に発見することはさらに重要である。多くの病院では定員に達しており、これらのチューブやラインを必要としている患者が増えています。カテーテルやラインの配置に関する迅速なフィードバックは、臨床医がこれらの患者の治療を改善するのに役立つ可能性がある。COVID-19以外にも、ラインやチューブの位置を検出することは、多くの病 院患者にとって常に必要なことである。<br> 

### data(DeepL)<br>
このコンテストでは、胸部レントゲン上のカテーテルやラインの存在と位置を検出します。機械学習を使用して、40,000枚の画像上でモデルを訓練してテストし、配置の悪いチューブを分類してください。<br> 

#### What files do I need?<br> 

トレーニング画像とテスト画像が必要です。このコンテストはコードのみのコンテストなので、隠れたテストセット（約4倍の大きさで14k枚の画像）もあります。<br> 
train.csvには画像ID、バイナリラベル、患者IDが含まれています。<br>
TFRecordsは訓練とテストの両方で利用可能です。(これらは隠れたテストセットでも利用可能です)。<br>
train_annotations.csvも含まれています。これらは、それを持つ訓練サンプルのセグメンテーションアノテーションです。これらは，競合他社のための追加情報としてのみ含まれています<br> 

#### Files<br>
- train.csv - 画像ID、バイナリラベル、患者IDが格納されています．
sample_submission.csv - 正しいフォーマットのサンプル投稿ファイルです．
- test - テスト画像
train - トレーニングイメージ

#### Columns<br>
- StudyInstanceUID - 各画像に固有のID
- ETT-異常-気管内チューブ留置異常
- ETT - 境界線 - 気管内チューブ留置境界線異常
- ETT - 正常 - 気管支内チューブの装着は正常です。
- NGT-異常-経鼻胃管留置異常
- NGT - 境界線 - 経鼻胃管留置境界線異常
- NGT - 画像化が不完全 - 画像化のために経鼻胃管留置が決定的ではない
- NGT - 正常 - 経鼻胃管留置は正常の境界線上にある。
- CVC - 異常 - 中心静脈カテーテル留置異常
- CVC - ボーダーライン - 中心静脈カテーテル留置境界異常
- CVC - 正常 - 中心静脈カテーテルの配置は正常です。
- スワンガンツカテーテルプレゼント
- PatientID - データセット内の各患者の一意のID

## Log<br>
### 20210223<br>
- join!!!<br>
- cassavaコンペのリベンジ。まだ3週間はあるので色々できるはず。<br>
- nb001(EDA)<br>
  - 自力でEDAを行った。
  - キャッサバコンペに比べて画像データがかなり大きい。512×512ぐらいに縮小してもいいとは思う。
  - 画像データは3チャネルあったが、3チャネル全てが同じであるため、平均をとって1チャネルの白黒画像にしても全く同じになる。それなら1チャネルだけの方が軽くていいと思う。<br>
  - 適当に閾値を設定して、それ以下の値のピクセルを全部0するのも良さそう。<br>
  - アノテーション画像を使うなら、画像データの縮小に合わせてアノテーション画像も縮小させる必要がある。(まだどう使えばいいのかよくわかってないけど...)<br>
  - クラスの割合が予想以上に不均衡だった。MoAコンペでは、ほとんどが0のクラスは、罰則を小さくすることしか学習できていなかったため、今回も気をつける必要があるかもしれない。<br>
  - とりあえず、trainingデータの平均でsample_submissionの表を埋めてsubmitしてみた。->LBは0.500だった。AUCだから当然か。<br>
- nb002(create_model_ResNext)<br>
  - ver1<br>
    - 再び、Y.NakamaさんのNotebookを写経。cassavaのときとほぼ一緒なのですぐ理解できた。汎用性が高いとはこういうことか。お陰様でいいスタートダッシュが切れた。感謝感謝。<br>
    - TrainDatasetまで書いた。いい復習になっている。<br>

### 20210224<br>
- [このNotebook](https://www.kaggle.com/raddar/errors-in-ett-abnormal-labels)を見ると、ETT -Abnormalのうちの一つがラベルが違うらしく、正確にはCVC - Abnormalらしい。EDAを見る限りでもそうっぽい。比較実験したい。<br>
- [このNotebook]の一番下のコメントで、timmとpytorch.xla_(だったっけ)のバージョンの齟齬の解消法が提案されてた。とりあえず、pytorch-image-modelsを使い続けて、xlaのバージョンを変えることにする。<br>
- どうやら、annotationを使うには、3stepでtrainingを行う必要があるらしい。<br>

- nb002<br>
  - ver2<br>
    - どうやら画像データのデータセットは自分で作らないといけないらしい。384×384で作ってから学習時に600×600にするのは、学習が早いかららしい。勉強のため、そうした場合と直接600×600に圧縮した場合の比較をしてみたい。恐らく、補完の際に若干崩れるため前者の精度は若干落ちると考えられる。<br>
    - 書き終わったので一旦quick saveした。<br>
  - ver3<br>
    - 公開Notebookとパラメータを同じにして動かした。<br>
    - ただし、時間短縮のため、foldは0のみにしてある。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9325 | 0.939 | 0.1152 | 0.1556 | <br>
    - CVはほぼ同じなので、恐らく再現できてる。
  - ver4<br>
    - ミスラベリングと議論されているデータを外した。また、全てのepochでモデルをセーブするようにした。<br>
      
- nb003(inference_ResNext)<br>
  - ver1<br>
    - 同じくY.NakamaさんのNotebookを写経した。<br>
    - 書き終わったので一旦quick saveした。<br>
  - ver2<br>
    - nb002_ver3の推論をしようとしたが、Internetを切り忘れた。<br>
  - ver3<br>
    - これ以降、trainingの方のNotebookに全て情報を書くため、このNotebookは登場させない。<br>
- nb004<br>
  - ver1<br>
    - 384×384の画像のデータセットを作るためのNotebook。nb002の元のNotebookにあったコメント通りにzipファイルにしてみた。<br>
    - 5GBもあるデータをいちいちローカルに落としてunzipして再びKaggleにあげ直すのは大変なので、Notebookの中でunzipして一つのファイルに入れるまでを行う。<br>
  - ver2<br>
    - unzipまで行った。<br>
    - データが取り出せない。なぜ。<br>
  - ver3<br>
    - 恐らく写真が多すぎるので諦めた。3stepの学習Notebookでは、画像サイズは普通に512だったため、そっちにする。<br>
- nb005(training_ResNeXt_step1)<br>
  - [Y,Nakamaさんのstep1]の写経。<br>
   - ver1<br>
     - TPUにとても苦戦する。よくわからんけど苦戦する。よくわからんから苦戦するのか。<br>
     - 一応全てのepochでモデルを保存するようにしておいた。<br>
     - 間違えてdebugをTrueにしたまま回してしまった。<br>
   - ver2<br>
     - debugをFalseに直した。<br>

### 20210225<br>
- nb002<br>
  - ver4<br>
    - ミスラベリングの疑いがあった、ETT - Abnormalクラスのデータを一つ外したところ、ETT - Abnormalクラスのaucが0.9336->0.9617と上がったので、恐らく指摘に間違いはない。その他のクラスのaucが若干下がってるのが気になるが、誤差としか考えられない。<br>
    - epoch5, 6を使うようにして、TTAを各5回にしたところ、LBが0.710まで落ちた。これは恐らく、学習の段階でaugmentationがRandomResizedCropとHorizontalFlipしかないことが原因だと思われる。<br>
    - TTAなどの変更は全て戻してsibmitした。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9308 | 0.941 | 0.1305 | 0.1551 | <br>
    - train_lossが高いのは、valid_lossが一番低いepochが手前にズレたから。<br>
    - やはり、ラベリングは間違っていたとみていいと思う。<br>
  - ver5<br>
    - Augmentationを増やしてみた。LB0.965を出している[このNotebook](https://www.kaggle.com/underwearfitting/single-fold-training-of-resnet200d-lb0-965)や、tawaraさんの[スレッド](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/210016)でもAugmentationを増やすと精度が上がっているため、恐らく間違いない。<br>
    - 1epochは最終層以外固定した。本当はこのように複数の変更を同時に行うべきではないが、時間がないので仕方がない。前回のコンペで効いているので入れる。<br>
    - どちらがより大きい影響を及ぼしているかがわからないが、学習が遅くなってる。もうちょいepochを増やせばCVはまだ上がりそう。このnbの工夫はそのままnb006あたりに生かしたい。<br>
  - ver6<br>
    - lossがうまく下がらなかった。**たとえ時間がなくても一気に二つ以上変更を加えてはいけない。これからは守る。**<br>
  - ver8(ver7は失敗)<br>
    - augmentationを多めに入れた。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9332 | 0.936 | 0.1312 | 0.1530 | <br>
    - CVはよくなってるけどLBが悪くなっている。<br>
  - ver9<br>
    - ver4から、1epochは出力層以外を凍結するように変更した。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9292 | 0.936 | 0.1189 | 0.1563 | <br>
    - どちらも悪化してしまった。Cassavaコンペで効いたことがこっちでも一様に効くわけではないらしい。<br>
- nb006(ResNeXt_step2)<br>
  - ver3(ver1, ver2は失敗)<br>
    - 動かした。<br>
- nb007(ResNeXt_step3)<br>
  - ver5(ver1はquick save, ver2, ver3, ver4は失敗)<br>
    - とりあえず写して動かした。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9530 | 0.951 | 0.1107 | 0.0.1359 | <br>
- nb009(公開Notebook)<br>
  - ver1<br>
    - 公開されている状態から、TTAを8回に増やした。<br>
    - その後に、inferenceの関数内でモデルを5fold分ロードする形にしてみたところ、実行速度がかなり遅くなった。ボトルネックは恐らくモデルを毎回ロードしなければいけないところだと考えられる。loaderがボトルネックだと勝手に思い込んでいたが、実は違うらしい。工夫次第では他のNotebookもさらに早くなるかもしれない。<br>
    - LBは0.955だった。TTAを無闇に入れてもダメなのか。aucとaccuracyでは勝手が違うので難しい。<br>
  - ver3(ver2)は失敗<br>
    - 公開Notebook通りに戻した。LBは0.965。それはそうだ。<br>
  - ver4<br>
    - vertical flipだけ入れてみた。<br>
    - LBは0.965だが、表示桁以下は若干上がっている。<br>
  - ver6(ver5は失敗)<br>
    - verticalとhorizontalを同時にひっくり返すものも入れた。<br>
    - 0.963に下がった。よくわからんなあ...<br>
  - ver8(ver7は失敗)<br>
    - ver4から、RandomResizedCropを入れた。<br>
    - LBは0.964だった。<br>
  - ver9<br> 
    - ver8から、ShiftScaleRotateを入れた。<br>
    - LBは0.964だった。この取り組み、意味あるのかな...<br>
    
### 20210226<br>
- nb002<br>
  - ver9(ver8は失敗)<br>
    - ver4から、augmentationを追加した。nb005のaugmentationにverticalflipを追加したもの。<br>

### 20210227<br>
- 予測確率を1.1乗したりすれば、aucがよくなったりするのかと思っていたが、aucに関係するのは予測確率の順序だけであるため、全く意味がなかった。aucを大きく誤解していた。反省。ということは、最初の方にやった、trainデータの平均で各列を埋めたサブミットはaucが0.5になって当然となる。(閾値を変えても、FPR=TPR=0とFPR=TPR=1にしかならないため、左下と右上の角を結ぶことになるため当然面積は0.5になる。)<br>
- nb009<br>
  - ver8(ver7は失敗)<br>
    - ver4から、RandomResizedCropを入れた。<br>
    - LBは0.964だった。<br>
  - ver9<br> 
    - ver8から、ShiftScaleRotateを入れた。<br>
    - LBは0.964だった。この取り組み、意味あるのかな...<br>
- nb010(training_ViT)<br>
  - ver1<br>
    - Cassavaコンペ3位のモデルを参考に、672x672の画像を9分割して224x224のViTにそれぞれ突っ込んで、attentionでweight averagingをするモデルを作ってみた。<br>
    - 全然ダメだった。特にCVCクラスがひどすぎる。画像を切ってしまうと複数の画像にカテーテルが跨ってしまうのがよくないんだろうな...<br>
  - ver2<br>
    - 画像を448×448にして、４等分にしてみたけど同じくダメだった。この案はボツ。<br>

### 20210301<br>
- 鳥蛙コンペの反省会にお邪魔させていただいた。てっきり音声データは全く異質なことをしているのかを思いきや割と画像コンペの側面が強かったよう。色々勉強させていただいた。
- nb011<br>
  - nb002_ver4をEfficientNetB3nsでやってみる。<br>
  - 伸びが明らかによくなかった。DiscussionでもEfficientNetを使っている人はほとんどいないようなので、大人しくResNet200Dを使おうと思う。<br>

### 20210302<br>
- Discussionを色々みた結果、一番強そうなのはやはりResNet200Dだった。また、4stage trainingが強そうだった。<br>
- annotationをどう使うかが今回の鍵になっていそう。一つの案としては、4stageにおける、3stageと4stageを交互に繰り返していくことがあげられる。そうすることで、よりteacherモデルの重みに近づけることができると予想。ちなみに、着想は鳥蛙コンペ5位のチームがやっていたcycle pseudo labelingから得た。<br>
- **ひらさんとマージした！！！ yukies爆誕！！！ 頑張るぞ〜**
- nb005<br>
  - ver5<br>
    - 4stageのモデルは画像サイズを640にしていたため、teacherモデルも画像サイズを640にした。<br>
- nb006<br>
  - ver4<br>
    - nb005_ver5の2stage。画像サイズを640x640にしたため、バッチサイズを16から8に落とした。本来は学習率も一緒に落とすが、そのままでもいけそうなのでそのままにした。epochを10に伸ばした。<br>
    - 少し動かした感じだと、学習率はやはり小さくするべきだった。あと、epochは10だと長すぎる。<br>
  - ver5<br>
    - 学習率を落として、epochも戻した。ほぼ公開Notebookのまま。やはり最初はそうしないと比較ができない。対照実験を守るべし。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: | 
      | 0.9380 | 2.0690 | 0.1601 | <br>
    - valid_lossが一番低いのがepoch2なのが気になる。学習率を落とした方がいいっぽい。<br>

### 20210303<br>
- annotationが間違ってる画像があるという意見がある。[このディスカッション](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/210064)に書かれていた。外した方がスコアが高い可能性もなくはない。<br>
- CVCのスコアはこのコンペの一つに鍵になっていると思う。ただ、[このディスカッション](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/216219)によると、CVCのAUCスコアが低いのはモデルが苦戦しているからではなく、CVCがほぼ全てのデータに入っているからだそう。[このディスカッション](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/207602)に外部データがある。使えるかも。<br>
- nb006<br>
  - ver6 ~ ver8<br>
    - 学習率を変える実験をしたので色々書く(ver5から載せる)<br>
    - | learning rate (ver) |  CV | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 2.5e-4(ver5) | 0.9380 | 2.0690 | 0.1601 | 
      | 1e-4(ver6) | 0.9454 | 1.7504 | 0.1573 | 
      | 5e-5(ver7) | 0.9486 | 1.7765 | 0.1492 | 
      | 3e-5(ver8) | 0.9507 | 1.9575 | 0.1360 |
      | 2e-5(ver10) | 0.9513 | 1.9946 | 0.1353 | 
      | 1e-5(ver9) | 0.9525 | 2.2128 | 0.1306 | 
    - 公開KernelのCVが0.9234であることを考えるとかなりよくなっている(一旦全てのstageを通過させたモデルなので単純比較はできないけど)。<br>
    - 学習率を落とすとepoch1のCVが高くなるが、それはもともとの重みが反映されているだけ。ここでの目的は親モデルの重みに近づけることであるため、train_lossもしっかり下がっているやつを選びたい。<br>
    - 2e-5, 1e-5に関しては上記の理由でepoch5のモデルを取っている。<br>
    - 以上から、ver10のモデルを採用する。<br>
    - ちなみに、この時点でsubするとLBは0.953だった。<br>
- nb007<br> 
  - ver9(ver6, ver7, ver8は失敗)<br>
    - nb006_ver10_epoch5のモデルを使う。ハイパラは全て同じ。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9582 | 0.954 | 0.0884 | 0.1318 | <br>
    - 学習率が大きく、完全に過学習している。<br>
    - 公開Notebookではvalid_lossが一番低いモデルが使われていたが、今回はCVが一番いいモデルを使った。[このディスカッション](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/222845)を見ると意見が割れているが、lossが小さい方を選ぶのは多数派で、AUCが良い方だけを見るとoverfitするらしい。今回は学習率が高すぎるので、次回以降に両方試してみる。<br>

### 20210304<br>
- 今実験しているサイクルに使っているモデルは、公開Notebookからそのまま取ってきただけのモデルであるため、今のfoldの切り方と同じ切り方で訓練した保証がなく、リークしている可能性がある。[このdataset](https://www.kaggle.com/ammarali32/startingpointschestx)で提供されている重みを使って、サイクル一周目からfold1を使ってモデルを作ろうと思う。そうしておいた方が後々全foldでモデルを作るときに手間も減る。
- nb005<br>
  - ver6<br>
    - fold1にした。また、重みの初期値を[この公開Dataset](https://www.kaggle.com/ammarali32/startingpointschestx/code)に変えた。<br>
    - 結果があまりよくなかった。そもそも、stage1での重みは変えなくて大丈夫だった。<br>
  - ver7<br>
    - 重みを戻した。<br>
    - ほぼCVが1まで上がり切ったので、こっちを採用する。<br>
- nb006<br>
  - ver11<br> 
    - fold1にした。ハイパラは全て同じ。<br>
    - 間違えてLB0.965のpretrainedを使ってしまっていた。<br>
  - ver12<br>
    - 子モデルの初期重みを直した。結果、重みを大きく学習しなければいけなくなったため、学習率を元の5e-4に戻すことにした。<br>
    - lossがかなり大きく、減りづらい。原因として、親モデルはImage Netの重みを使っているのに対して子モデルは医療画像用の重みを使っていることが考えられる。親モデルも後者にすべきだと考えたので、nb006_ver6の重みに変更してみる。<br>
    - 学習率を2e-5に戻した。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: |
      | 0.9272 | 1.7101 | 0.1547 | <br>
    - valid_lossが一番低かったepochを書いた。<br>
  - ver13<br>
    - 学習率を1e-5にした。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: |
      | 0.9092 | 1.8908 | 0.1717 | <br>
    - 悪化したので、2e-5を採用する。<br>
- nb007<br>
  - ver9, ver10, ver11, ver12, ver13<br>
    - 学習率を変える実験を行った。<br>
    - | learning rate (ver) | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | :---: |
      | 2.5e-4(ver9) | 0.9582 | 0.954 | 0.0884 | 0.1318 | 
      | 1e-4(ver10) | 0.9614 | 0.961 | 0.1142 | 0.1222 | 
      | 5e-5(ver11) | 0.9670 | 0.964 | 0.1157 | 0.1126 | 
      | 2e-5(ver12) | 0.9708 | 0.966 | 0.1002 | 0.1058 |
      | 1e-5(ver13) | 0.9711 | 0.965 | 0.0955 | 0.1041 | <br>
    - 1e-5と2e-5ではほとんど差がなかった。nb006と同じ2e-5を採用することにする。<br>

### 20210305<br>
- [Focal Lossの原論文](https://arxiv.org/pdf/1708.02002)と[Qiita記事](https://qiita.com/agatan/items/53fe8d21f2147b0ac982)に目を通した。実装は[鳥蛙コンペのディスカッション](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/212978)にあった。<br>
- 他の論文で、AUCを微分可能な関数で近似して直接最大化する手法が紹介されていた。メラノーマコンペなどで有効性が示されているらしい([論文のリンク](https://arxiv.org/pdf/2012.03173))。実装してみたいがあまりに大変そうなので後回しにする。<br>
- nb005<br>
  - ver8<br>
    - 医療画像のpretrainedでfold2の親モデルを作った。<br>
- nb007<br>
  - ver14<br>
    - Focal Lossを試す。とりあえず、alpha=1(alphaの設定の仕方が原論文と違うが、alpha=1は、alphaは使用しないのと等価)、gamma=0.1にした。<br>
    - ミスラベルの疑いがあるデータを外すことを忘れていたことに気がついた。これだけのために対照実験するのはあまりにも手間であることに加え、本来は外すべきものだったため、外しておく。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9708 | 0.964 | 0.0927 | 0.1053 | <br>
    - LBが若干下がった。逆効果なのかもしれない。<br>
    - epoch4の途中でlossが急にnanになってしまい、ストップしてしまった。Focal Lossは数値的な安定性に若干かけている印象を受ける。実装の仕方の問題であろうが、注意しなければならない。<br>
  - ver15<br>
    - gammaを0.5にした。<br>
    - ローカルのGPUを使ったため、バッチサイズが8になって、学習率が1e-5になっている。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9709 | 0.966 | 0.1006 | 0.1085 | <br>
    - LBはほぼ同じだが、表示桁以下は下がっている。僅差であるため、リークが完全にないモデルを作ってからもう一度試す価値はありそう。
    - valid_lossが一番低いモデルとCVが一番高いモデルを合わせてみたが、特に意味はなかった。伸びたとしても表示桁以下の違いにしかならなさそう。<br>
      
### 20210306<br>
- 最初からサイクルの2周目に色々な工夫を入れようとしたが、まずはサイクルの2周目でモデルの精度が改善されるかを確認する方が先だと思ったのでそうする。別の工夫は、モデルを作る段階(2・3stageの一周目)で取り組む。
- nb006<br>
  - ver14<br>
    - ver12から、ResNet200dにDual Attentioinを追加した。[参考にしたNotebookのリンク](https://www.kaggle.com/jy2tong/efficientnet-b2-soft-attention)。<br>
    - メモリに乗り切らなかったため、バッチサイズを8に落としたが、学習率を変え忘れてたのでボツ。<br>
  - ver15<br>
    - バッチサイズを16->8にして学習率を 2e-5 -> 1e-5 に変えた(**前はバッチサイズが16だったと勘違いしていたが、実はずっと8だった。なので、これからも学習率は1e-5にする**)。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: |
      | 0.9505 | 1.8525 | 0.1333 | <br>
    - めちゃめちゃ強い。Publicの完成品をfine tuneした時と変わらない精度が出てる。<br>
  - ver16<br>
    - SAMを導入した。そのため、scalerは外している(apexを使いながらSAMを使う方法がわからなかった)。<br>
    - foldごとにモデルを全部セーブする方式をやめた。<br>
    - 学習率を1e-5にしていたけど、SAMに限ってはなかなか学習が進まないので止めた。<br> 
  - ver17<br>
    - nb007_ver15.5のモデルの2回目のstage2。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: |
      | 0.9534 | 1.7517 | 0.1366 | <br>
  - ver18(ver17は失敗)<br>
    - 学習率を2e-5にした。<br>
    - Lossが大きいのが気になって途中で止めてしまった(止めなければよかった)。
- nb007<br> 
  - ver15(ver15.5)<br>
    - nb006_ver12のモデル(リークがない、fold1のやつ)を学習させた。<br>
    - 今気づいたが、ver15が二つある。これはver15.5ということにする。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9635 | 0.961 | 0.1075 | 0.1273 | <br>
  - ver19(ver16, ver17, ver18は失敗)<br>
    - nb006_ver15のモデルのfine tune。Dual Attentionをつけた。<br>
    - どのモデルでも後半は過学習しているので、epochを5から4に変更する。<br>
    - なぜかエラーを吐かれてしまった。<br>
  - ver20<br>
    - ひらさんに回してもらうためにquick saveした。<br>
  - ver21<br>
    - ver15.5と同じ。サイクルの２週目を回す。ミスラベルのデータを外すのを忘れてた(またやっちまったけどしゃーない...)。<br>
    - epoch4にするのを忘れていた。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9679 | 0.963 | 0.1077 | 0.1185 | <br>
    - 良い感じに伸びた。仮説は当たっていたらしい。<br>

### 20210307<br>
- nb006<br>
  - ver19<br>
    - 学習率を3e-5にした。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: |
      | 0.9346 | 2.2481 | 0.1489 | <br>
    - epoch5で過学習してしまっている。<br>
  - ver20<br>
    - 結局学習率を2e-5にした(何してんだよ俺)。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: |
      | 0.9254 | 2.3515 | 0.1561 | <br>
    - 学習率は3e-5の方がよかった。<br>
  - ver21<br>
    - ver12にFocal_Loss(gamma=0.5を入れた)。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: |
      | 0.9316 | 1.6560 | 0.1679 | <br>
      
- nb007<br>
  - ver23(ver22はquick save)<br>
    - これもquick save。ひらさんにDANetのリベンジをしてもらうべく、ver19のバッチサイズを12にして、epochを5に戻した(自分のNotebookでは14になっているが、ひらさんに回してもらうときに12になった)。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 0.9648 | 0.963 | 0.1063 | 0.1223 | <br>
    - まだvalid_lossが下がり続けているため、もっといける気がする。<br>
  - ver24<br>
    - [ディスカッション](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/212532)で、trainingの最後でaugmentationを外すと精度が改善される言われている。[原論文のリンク](https://arxiv.org/pdf/1909.09148.pdf)。ということで、lr=1e-6, min_lr=5e-7, epoch=2にして回してみることにした。<br>
    - 元のモデルは、ver21のfine tuneモデル。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 0.9687 | 0.963 | 0.0961 | 0.1142 | <br>
    - CVは若干上がっている。LBも多分、表示桁以下は上がっていると思う。<br>
  - ver25<br>
    - nb006_ver21のFocalLossモデル。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 0.9632 | 0.962 | 0.1229 | 0.1221 | <br>
    - 一応LBは改善している。工夫の一つとして使えそう。<br>

### 20210308<br>
- nb006<br>
  - ver22<br>
    - nb007_ver21のfine tuningモデルを、さらにfine tuningする。設定はver13と全く同じ。nb007でFocalLossも使う予定。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: |
      | 0.9535 | 1.472 | 0.1369 | <br>
  - ver23<br>
    - nb007_ver30のモデルの重みを使って、DANet moduleをつけてpretrainした。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9552 | 1.4004 | 0.1284 | <br>
    - nb007_ver30の方のモデルのLBが0.959で異様に低かったので、こちらのモデルの使用も見送ることにした。<br>
- nb007<br>
  - ver26<br>
    - ver23のDANet moduleモデルを、バッチサイズ14で回す。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 0.9656 | 0.964 | 0.1055 | 0.1214 | <br>
    - 若干だけどCVがよくなって、LBも0.001上がっている。まだまだ上がりそうなので、ver25のfine tuningを試してみたい。<br>
  - ver27<br>
    - ver24からroc_star_lossを追加する。roc_starは、Deep AUCと同じような、AUCを最大化することを狙った損失関数。[GitHubのリンクはここ](https://github.com/iridiumblue/roc-star)で、参考にしたNotebookは[ここ](https://www.kaggle.com/iridiumblue/roc-star-an-auc-loss-function-to-challenge-bxe/log)<br>
    - GPUで動作確認をしたので、一旦quick saveした。<br>
  - ver28<br>
    - ver24のforkで、min_lr=9e-6にした。epochも3にした。モデルはnb007_ver26のもの。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 0.9664 | 0.965 | 0.0984 | 0.1190 | <br>
    - CV, LBともに若干改善した。しっかり効果が出ている。<br>
  - ver29<br>
    - ver27を回した。モデルはver28と同じ。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 0.9664 | 0.963 | 40.5113 | 0.1187 | <br>
    - roc_star_lossを使っているため、train_lossの値は他とは異なる。<br>
    - 学習時間もかなり長くなった上にスコアも落ちてしまった。これは却下する。<br>
  - ver30<br>
    - nb006_ver21のモデルを、FocalLossを用いてfine tuningする。設定はver15と同じ。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 0.9697 | 0.959 | 0.1051 | 0.1055 | <br>
    - どうしてLBがこんなに低いのかさっぱりわからない。これならうまくいくと思っていたのに...<br>

### 20210309<br>
- nb007<br>
  - ver31<br>
    - ver30から、FocalLossのgammaを0.1に変える。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 0.9689 | 0.957 | 0.0983 | 0.1034 | <br>
    - 見るに耐えないのでFocal Lossは一旦保留にする。何かがおかしい可能性は否定できないけど、時間もない。<br>
  - ver32<br>
    - nb006_ver22のモデルのstage3を行った。cycleの３周目。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: | 
      | 0.9658 | 0.1075 | 0.1200 | <br>
    - 悪化してしまった。Focal Lossが全部ダメだったのもこのせいである可能性が高い。<br>

### 20210310<br> 
- nb006<br>
  - ver24<br>
    - nb007_ver21のモデルにDANet moduleを使ってpretrainする。nb006_ver15を使って、参照先をnb007_ver21のモデルに変える。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9518 | 1.4768 | 0.1323 | <br>
    - あまりver15からの改善は大きくはない。<br>
- nb007<br>
  - ver33<br>
    - 前回はcycle の３周目のモデルに対してFocal Lossを適用してスコアが下がったが(ver30)、そもそも、Focal Lossを使わなくてもスコアが下がっていることがわかった(ver32)。なので、cycleの2周目のモデル(nb006_ver17)を使って、nb007_ver30で学習させてみる。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 0.9672 | 0.959 | 0.1056 | 0.1109 | <br>
    - FocalLossは何をやっても効かないのて完全にボツにする。<br>

### 20210311<br>
- nb006<br>
  - ver25<br>
    - DANet　moduleをつけた状態でもう一度step2とstep3を学習させてみることにした。そのstep2。ver15とハイパラは同じで、モデルの初期値がnb007_ver26になっている。<br>
    - | CV | train_loss | valid_loss |
      | :---: | :---: | :---: | 
      | 0.9590 | 1.6305 | 0.1301 | <br>
    - めちゃくちゃ強い。これが最強説ある。<br>
- nb007<br>
  - ver34<br>
    - nb007_ver26を、nb006_ver24のモデルに対して適用する。cycle x2 -> DANet module<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: | 
      | 0.9463 | 0.963 | 0.1110 | 0.1240 | <br>
    - 悪化してしまった。この方向性はよくないので変える。<br>
- nb014(SeResNet152D-step1)<br>
  - ver1<br>
    - nb005_ver6のモデルをSeResNet152Dに変更した。デバイスをGPUにしたので、バッチサイズが16 -> 8に学習率が5e-4 -> 2.5e-4に、変わっている。<br>
    - モデル名を変え忘れた。中身はちゃんと変わってる。<br>
- nb015(EfficientNetB5ns-step1)<br>
  - ver1<br>
    - nb005_ver6のモデルをEfficientNetB5nsに変更した。デバイスをGPUにしたので、バッチサイズが16 -> 8に学習率が5e-4 -> 2.5e-4に、変わっている。<br>
- nb016(SeResNet152D_step2)<br>
  - ver1<br>
    - nb006_ver12のモデルをSeResNet152Dに変えた。<br>
  - ver2<br>
    - ver1の親モデルがGPUに乗っていなかったので、乗せた。意味なかった。<br>
  - ver3<br>
    - nb006_ver15と同じで、DANet moduleをつけた。ローカルで回すとギリギリOOMになったのでバッチサイズを7に下げた。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9407 | 3.1717 | 0.1377 | <br>
    - ResNet200Dに比べると若干弱い。ただ、学習率をもうちょい上げればまだCVは伸びそう。<br>
  - ver4<br> 
    - batch_sizeを4にして、gradient_accumulationを2にした。<br>
    - [このページ](https://kenbell.hatenablog.com/entry/2020/01/26/134350)によると、gradient_accumulationをやっても、Batch_Normalization層はbatch１つ分しか認識しない(確かにそれはそう)ため、完全に再現できるわけではないそうだ。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.8921 | 3.6073 | 0.1562 | <br>
    - 確実に悪化してる...<br>
  - ver5<br>
    - gradient_accumulation=4にしてlr=2e-5にしたが、途中経過があまりにも悪すぎて途中で止めた。<br>
  - ver6, ver7<br>
    - batch_size=7に戻して、optimizerをRAdamとAdaBeliefに変えた。Adamの結果と合わせて記載する。<br>
    - | optimizer(version) | CV | valid_loss | 
      | :---: | :---: | :---: | 
      | Adam(ver3) | 0.9407 | 0.1377 | 
      | RAdam(ver6) | 0.9376 | 0.1426 |
      | AdaBelief(ver7) | 0.9439 | 0.1402 | <br>
  - ver8<br>
    - optimizerをAdamに戻して、batch_size=7のままgradient_accumulation=2に、lr=2e-5してみた。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: |
      | 0.1447 | 3.3681 | 0.1447 | <br>
    - もはやgradient_accumulationって悪影響しか与えない結果になってしまった。実装は間違っていないように見えるのでおかしい気もするが諦める。<br>
- nb017(EfficientNetB5ns_step2)<br>
  - ver1<br>
    - nb006_ver12のモデルをEfficientNetB5nsに変えた。<br>
  - ver2<br>
    - nb016_ver3のモデルをEfficientNetに変えた。こっちの方がメモリ消費が若干激しいので、nb016の方でgradient_accumulationの実験をしてからこっちを動かすことにする。<br>
    - nb016の方で何をやってもうまくいかなかったので、普通にAdamでbatch_size=6, gradient_accumulation=1で回す。<br>

### 20210313<br>
- nb007<br>
  - ver35<br>
    - nb006_ver25(DANet module付、2周目)のモデルを学習させた。nb007_ver26からバッチサイズを15に上げた。<br>
  - ver36, ver37<br>
    - lrを下げた。パラメータ探索を軽くやりたいため、1epochのみで止めた。<br>
    - | lr(version) | CV | valid_loss |
      | :---: | :---: | :---: |
      | 2e-5(ver35) | 0.9640 | 0.1271 |
      | 1e-5(ver36) | 0.9642 | 0.1254 | 
      | 5e-6(ver37) | 0.9638 | 0.1247 | <br>
    - CVはほぼ全部同じであるため、lossの下がり方と数値を見て5e-6を採用する。<br>
  - ver38<br>
    - ver35のlrを5e-6にした。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9651 | - | 0.1052 | 0.1224 | <br>
    - 結局nb007_ver26に勝てない...<br>
  - ver39<br>
    - ver38のモデルをwoaug_fine_tuningする。ハイパラはnb017_ver4と同じ。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9658 | 0.965 | 0.0956 | 0.1210 | <br>
    - LBは変わらなかったし、step2とstep3を一回ずつだけのモデルの方がvalid_lossは低い。失敗した...<br>
- nb016<br>
  - ver9<br>
    - ver4からbatch_size=8にした。<br>
  - ver10<br>
    - AdaBeliefに変えた。<br>
    - 二つの実験結果をまとめる。<br>
    - | optimizer(ver) | CV | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | Adam(ver9) | 0.9470 | 3.2392 | 0.1354 | 
      | AdaBelief(ver10) | 0.9420 | 3.3278 | 0.1387 | <br>
    - Cassavaでも今回でもAdamの方がよかった。１からモデルを作る場合とfine tuningをする場合ではまた違う結果になるのだろうか。理由がいまいちよくわからない。<br>
  - ver11<br>
    - woaug_fine_tuningをやってみる。nb017_ver4とハイパラは同じ。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9485 | 3.0327 | 0.1317 | <br>
- nb017<br>
  - ver3<br>
    - nb016_ver9をモデルをEfficientNetB5nsに変えた。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9439 | 0.7462 | 0.1386 | <br>
    - なんかまだ下がりそうだな。<br>
  - ver4<br>
    - TPUが空いていなくてGPUに余裕があるので、wo aug fine tuningを試してみる。lr=1e-6, min_lr=5e-7, epochs=3, T_max=3にした。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9490 | 0.7000 | 0.1347 | <br>
- nb018(SeResNet152D)_step3<br>
  - ver1<br>
    - nb007_ver26をモデルをSeResNet152Dに変えてnb016_ver9のモデルを学習させた。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9609 | - | 0.1250 | 0.1282 | <br>
  - ver2<br> 
    - nb016_ver11のモデルを学習させた。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9607 | - | 0.1248 | 0.1285 | <br>
    - ほぼ変わらなかったので、ver1を使う。<br>
  - ver3<br>
    - ver1のモデルでwoaug_fine_tuningを行った。
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9628 | 0.946 | 0.1054 | 0.1266 | <br>
    - なんでだよーーーーーーーーーーーーーーー<br>
- nb019(EfficientNetB5ns_step3)<br>
  - ver1<br>
    - TPUでどうしても動かないなあと思っていたが、よく考えたらtimmのEfficientNetはTPUだとうまく動かないとCassavaのときから言われていた。そこで、GPUで動かす。batch_sizeを16から8に落としてlrを2e-5から1e-5に下げた。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9615 | 0.1343 | 0.1252 | <br>
    - valid_lossの極小がepoch1にきてしまったため、明らかにlrが高すぎた。woaug_fine_tuningで調整すればいいか。<br>
  - ver2<br>
    - ver1のモデルをwoaug_fine_tuningした。epochは3にした。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9629 | 0.939 | 0.1086 | 0.1232 | <br>
    - LBおかしくね？なんで？？？
- nb020(DenseNet121_step1)<br>
  - ver1<br>
    - fold1で、DenseNet121の親モデルを作った。<br>
    - ETT - Abnormalが(他のモデルもそうだけど、一番)極端に悪い。流石にもうちょい上げたい。<br>
  - ver2<br>
    - 勘で、epochs=5, T_max=5, lr=1e-4にした。<br>
    - best_scoreのモデルがいい感じになったので、それを採用する。<br>
- nb021(DenseNet121_step2)<br>
  - ver1<br>
    - nb017_ver3のモデルをDenseNet121に変えた。ハイパラなどは全て同じ<br>
    - ETT - AbnormalクラスのAUCが明らかにおかしいので止めた。<br> 
  - ver2<br> 
    - 親モデルをbest_lossに変えた。<br>
    - CVが0.85ぐらいまでしか上がらなくて明らかにダメそうなのでここで諦める。<br>
### 20210314<br>
- 半分以上昨日のところに書いてしまった。<br>
- nb006<br>
  - ver26<br>
    - nb006_ver15のモデルをwoaug_finetuningした。<br>
    - スコアは若干上がったけどlossが悪化したので止める...<br> 
  - ver28(ver27は失敗)<br>
    - 0.965のPublicモデルにDANet moduleをつけた。foldは0にしてある。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9531 | 1.6474 | 0.1202 | <br>
    - 強い。<br> 
- nb007<br>
  - ver40<br>
    - nb006_ver28のモデルのstep3。ver26をforkしてlr=1e-5に、batch_size=15に変更した。<br> 
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9668 | 0.0907 | 0.1078 | <br>
    - 過去最高級に強い。<br>
### 20210316<br>
- nb005<br>
  - ver9<br>
    - [ひらさんが見つけてきてくれたfoldの切り方](https://www.kaggle.com/underwearfitting/how-to-properly-split-folds)にしたがってfoldを切った。fold0を使う。親モデルを作った。<br>
    - 最初からDANet moduleをつけてみた。<br>
    - lossが一番低いepochのCVが、ETT - abnormalクラスだけ異常に低いが、そもそもかなり陽性のデータが少ないので気にしない。<br>
- nb006<br>
  - ver29<br>
    - nb005_ver9の親モデルを使って子モデルを作った。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9459 | 0.3231 | 0.1213 | <br> 
    - train_lossの下がり方が全然違った。重みを近づけるなら、そもそもモデルの形を全て揃えるべきだったと考えられるし、foldも最初からこの切り方をすべきだった。<br>
    - このepochはたまたまCVが低いが、直前では0.956まで上がっているので問題はないと思う。それよりlossの下がり方がすごい。<br>
- nb007<br>
  - ver41<br>
    - ver40のモデルのwoaug_fine_tuning。ひらさんに回してもらう。<br>
    - | CV | LB | train_loss | valid_loss | 
      | :---: | :---: | :---: | :---: | 
      | 0.9673 | 0.962 | 0.0779 | 0.1082 | <br> 
    - LB低い。なんもわからん。<br>
    - 後から気づいたが、augmentationを外し忘れていた。まあどのみち結果はあんまり変わらなかった気がする。<br>
  - ver42<br>
    - nb006_ver29のモデルの学習を回す。ひらさんに頼んだ。<br>
    - | CV | LB | train_loss | valid_loss | 
      | :---: | :---: | :---: | :---: | 
      | 0.9641 | 0.964 | 0.1109 | 0.1135 | <br>
    - これは期待できそう。woaug_fine_tuningもやる。<br>
  - ver43<br>
    - ver42のモデルのwoaug_fine_tuning。<br>
    - ひらさんが回す直前Datasetを差し替えてしまい、失敗した。<br>
- nb016<br>
  - ver12<br>
    - 公開Notebookの重みにDANet moduleをつけて学習させた。<br>
    - | CV | train_loss | valid_loss | 
      | :---: | :---: | :---: | 
      | 0.9460 | 3.0299 | 0.1252 | <br>
    - 当然強いけど、nb007_ver42の方が優先かなあ<br>
- nb018<br>
  - ver4, ver5<br>
    - nb016_ver12のモデルのstep3をひらさんに回してもらおうとしたが、エラーがでて断念した。<br>
  - ver7(ver6は失敗)<br>
    - SeResNetのPublic Weightをfine tuningしたが、foldの切り方をPatient leakが起こらないものにした結果、逆にリークが大きくなったらしくCVが異常に高くなったしまった。<br>
  - ver8<br>
    - foldの切り方を戻してひらさんに回してもらった。<br>
    - | CV | LB | train_loss | valid_loss |
      | :---: | :---: | :---: | :---: |
      | 0.9683 | 0.962 | 0.1133 | 0.1087 | <br> 
    - 結局リークしてるけどもう時間がない。これ本当は絶対に良くないやつだよな...反省...<br>
    - 表示桁以下はスコアが改善しているものと信じる(後から確認したら0.0004ぐらい上がってた)。<br>

### 20210316<br>
- もうここに記録するのを怠り始めていて、何が何だかだんだんわからなくなってきている。<br>
- nb007<br>
  - ver44<br>
    - ver43のwoaug_fine_tuning。<br>
- nb009<br> 
  - ver15<br>
    - 公開Notebookのモデルをほぼ全て差し替えた。<br>
      - ResNet1はTTAにvertical flipを追加。<br>
      - ResNet2はnb007_ver12に差し替え(LB0.966)
      - ResNet3としてDual Attention Moduleがついた自作ResNetを追加(LB0.965)。<br>
      - SeResNet152Dはfine tuningしたnb007_ver8に差し替えた。<br>
      - PublicのEfficientNet、Multi Headを拝借した。本当は全部自力で行きたかった。力不足...<br>
    - 何はともあれLBが0.968で銅圏に入れた。<br>
