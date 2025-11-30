# 2025

## 基礎実習1

### Python基礎

[こちらのリンクから](https://colab.research.google.com/drive/1xXIwb8mUwa3uT0cQ4bZH5n6NDu5iO17B?usp=sharing)

<!-- 
[解答](https://colab.research.google.com/drive/1i3UjtGQZSJikDa6v4PpfFYrPkyISHC_1?invite=CI7liv0J)
-->
### 画像認識

GitHubのレポジトリをクローンする


```bash
!git clone https://github.com/rits-menglab/2025_practical_application
```

ディレクトリを移動する

```bash
cd 2024_practical_application
```

ImageNetのクラス情報(JSON)をダウンロード

```bash
!wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
```

JSONファイルを読み込む

```python
import json
class_index = json.load(open('imagenet_class_index.json', 'r'))
print(class_index)
```

辞書のキーをstring型からint型へ変換

```python
labels = {int(key):value for (key, value) in class_index.items()}
print(labels[332])
```

ImageNetで学習済みの画像認識モデルを読み込む

```python
from torchvision import models
from utils import *

model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).eval()
```

画像を読み込み、テンソルデータに変換

```python
img_path = './sample_data/sample1.jpg'
tensor_img = make_tensor_img(img_path)
```

モデルに入力し、出力を表示

```python
out = model(tensor_img)
predict = out.argmax(-1)
for i in predict:
    print(labels[i.item()][-1])
```
## 基礎実習2

### 物体検出

GitHubのレポジトリをクローンする

```bash
!git clone https://github.com/ultralytics/yolov5
```

ディレクトリを移動する

```bash
cd yolov5
```

必要なパッケージをインストール

```bash
!pip install -r requirements.txt
```

サンプル画像の確認

```python
from PIL import Image
import os
import glob
paths = glob.glob(os.path.join('data','images','*.*'))
imgs = []
for img in paths:
    imgs.append(Image.open(img))
imgs[0]
```

学習済みの物体検出モデルを用いて、画像を推論

```bash
!python3 detect.py
```

結果の画像を表示  
結果画像のパスは **「Results saved to {結果画像のパス} 」** と表示される

```python
paths = glob.glob(os.path.join('{結果画像のパス}','*.*'))
imgs = []
for img in paths:
    imgs.append(Image.open(img))
imgs[0]
```

### 画像生成１

[こちらのリンクから](https://colab.research.google.com/drive/1dhKHkm3qHYfWKjmUERNgmRvJxKwXS4lM?usp=sharing)

### 自然言語処理

[こちらのリンクから](https://colab.research.google.com/drive/1wazNe_v5AnYnSAeYV4dfGz6XxH2oWgXz?usp=sharing)

### 画像分割
[こちらのリンクから](https://colab.research.google.com/drive/1D6InfWOwsKNsE9jL9oJC6vaUPR-Ovd_p?usp=sharing)

### パーセプトロン

[こちらのリンクから](https://colab.research.google.com/drive/188BM4B5aAk1t2le7w-uPpQy_ORlTwES-?usp=sharing)

## 基礎実習3

### 画像生成２

[こちらのリンクから](https://colab.research.google.com/drive/1sBGFQpqCeAVJ54Pt7B_o88QwT9c-a5m3?usp=sharing)

### SegGPT

[こちらのリンクから](https://colab.research.google.com/drive/1ajqgEDAT19vBg2Wzyd18PLndheEJ4CG5?usp=sharing)

### 軽量化

[こちらのリンクから](https://colab.research.google.com/drive/1cCZwh0MB8txkFgIlFifi5vQm1Tk9F7gS)

### 畳み込みニューラルネットワーク

[こちらのリンクから](https://colab.research.google.com/drive/1Sgi3Ic3vMp30au0rNjh96KrZP-pFxNHA?usp=sharing)

### 強化学習1

[こちらのリンクから](https://colab.research.google.com/drive/1XuMYlvz38LbsM1mnipjXA72TnbDpFgYB?usp=sharing)

### 強化学習2

[こちらのリンクから](https://colab.research.google.com/drive/1f2lNmsQOl_ECVy27csxu9nMRzzLHepaE?usp=sharing)

## 研究体験1

### 手書き文字認識

[こちらのリンクから](https://colab.research.google.com/drive/1pW1VZmzsojO4F1jlX7K_9InAy1E8Tdz0?usp=sharing)

## 研究体験2

### 異常検知

[こちらのリンクから](https://colab.research.google.com/drive/1AmZH6W5Hcefy-6eb9IFY-8GqDDFTHeL_?usp=sharing)

