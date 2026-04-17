'''
@echo off
set PROJECT_DIR=%~dp0
cd /d %PROJECT_DIR%\..

echo --- Shogi Engine Search Selection ---
echo [1] MCTS (Monte Carlo Tree Search)
echo [2] NPLS (Neural Priority List Search)
set /p CHOICE="Select Search Type (1 or 2, default=1): "

set SEARCH_TYPE=mcts
if "%CHOICE%"=="2" set SEARCH_TYPE=npls

echo starting with %SEARCH_TYPE%...
echo.

echo --- Start GNN Shogi Engine ---
:: 仮想環境のPythonで実行
".\venv\Scripts\python.exe" main.py --search_type %SEARCH_TYPE%

echo.
echo --- Engine Stopped ---
pause
'''

'''
NPLS(Neural Priority Leaf Search)
MCTS(Monte Carlo Tree Search)
'''

'''
# UW-NPLS（Uncertainty-Weighted Neural Priority Leaf Search）

## 概要

UW-NPLSは、従来のMonte Carlo Tree Search（MCTS）をベースにした探索とは異なり、**逆伝播（バックプロパゲーション）を用いず、葉ノード集合を優先度付きで直接探索する新しい探索アルゴリズム**です。

さらに、囲碁AIで実績のある「不確実性に基づく探索資源配分」の考え方を取り入れ、

> **「有望だが不確実な手を重点的に読む」**

という戦略を実現しています。

これにより、探索の**高速化・安定化・精度向上**を同時に達成します。

---

## 背景と目的

従来のMCTSは高い性能を持つ一方で、以下の課題があります。

* 逆伝播による計算コストが高い
* 木構造の更新がボトルネックになる
* GPU並列化が難しい
* ニューラルネットの誤評価に弱い

UW-NPLSはこれらを解決するために設計されています。

---

## 基本アイデア

UW-NPLSでは、探索対象を「木」ではなく、

> **優先度付き葉ノード集合（Priority Queue）**

として管理します。

探索は以下のように進みます：

1. ルート局面を評価
2. 優先度が最も高いノードを選択
3. そのノードを展開して子ノードを生成
4. 子ノードを優先度付きキューに追加
5. 制限時間またはノード数まで繰り返す

---

## 優先度関数（核心）

各ノードの優先度は以下で定義されます：

```
Priority =
  α * Q   （評価値）
+ β * P   （方策確率）
+ γ * D   （深さ補正）
+ δ * U   （不確実性）
+ ε * E   （探索ボーナス）
+ ζ * UW  （不確実性強化ボーナス）
```

---

### 各要素の意味

* **Q (Value)**
  ニューラルネットによる局面評価（勝率）

* **P (Policy)**
  方策ネットによる手の確率

* **D (Depth)**
  深い読みを優遇する補正（logスケール）

* **U (Uncertainty)**
  モデルの不確実性（policyエントロピーなど）

* **E (Exploration)**
  未探索ノードを試すためのボーナス

* **UW（Uncertainty-Weighted Bonus）**

  ```
  UW = U / (1 + sqrt(visit_count))
  ```

  不確実なノードを優先的に探索するための重要な項

---

## 不確実性を利用した探索（UWの役割）

UW-NPLSの最大の特徴はここです。

通常の探索：

```
評価が高い手 → 重点的に探索
```

UW-NPLS：

```
評価が高い かつ 不確実 → さらに重点的に探索
```

つまり、

> **「怪しいが強い可能性のある手」を見逃さない**

という探索が可能になります。

---

## 初手選択（Root Move Aggregation）

探索結果は単一ノードではなく、初手ごとに集約されます。

```
Score =
  w1 * 最大評価
+ w2 * 平均評価
+ w3 * 探索回数
- w4 * 分散（不安定さ）
```

---

### 分散ペナルティの効果

* 評価が安定している手を優先
* 一時的に高評価な手（ノイズ）を排除

例：

| 手 | 平均評価 | 分散  | 結果         |
| - | ---- | --- | ---------- |
| A | 高い   | 大きい | 不安定        |
| B | やや高い | 小さい | 安定（選ばれやすい） |

---

## 主な特徴

### 1. 高速性

* 逆伝播不要
* 木構造更新なし
* GPU並列化に適している

---

### 2. 高精度

* 不確実性を考慮した探索
* NN誤差への耐性向上

---

### 3. 高安定性

* 分散ペナルティによる安全な意思決定

---

### 4. 探索の柔軟性

* 深い読み筋（詰み・攻め合い）に強い
* tacticalな局面に強い

---

## 特に有効な局面

UW-NPLSは以下のような局面で特に強力です：

* 詰みや必至が絡む局面
* 攻め合い・寄せ合い
* 評価が拮抗している局面
* NNが迷いやすい局面

---

## パラメータ

| パラメータ       | 意味            |
| ----------- | ------------- |
| alpha       | 評価値の重み        |
| beta        | 方策の重み         |
| gamma       | 深さ補正          |
| delta       | 不確実性          |
| epsilon     | 探索ボーナス        |
| zeta        | 不確実性強化（UW）    |
| var_penalty | 初手選択時の分散ペナルティ |

---

## まとめ

UW-NPLSは、

> **高速なBest-First探索 + 不確実性駆動探索 + 安定化機構**

を統合した新しい探索アルゴリズムです。

従来のMCTSと比較して、

* より高速に
* より柔軟に
* より安定した判断を行う

ことが可能になります。

---

## 今後の拡張

* GPUバッチ探索（高速化）
* valueの分散による不確実性強化
* 詰み探索とのハイブリッド
* 不完全情報ゲームへの拡張

---

UW-NPLSは、将棋AIに限らず、チェス・囲碁・カードゲームなど幅広い分野への応用が可能な汎用探索フレームワークです。

'''
