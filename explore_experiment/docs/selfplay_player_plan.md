# 将棋自己対局用 Player 実装方針（調査メモ）

## 目的
- `explore_experiment` に、強化学習向けの自己対局データ生成専用 `player` を追加する。
- 1対局ごとに `(state, policy_target, value_target)` を保存し、既存学習系（`train/train.py`）へつなげる。

## 調査サマリ（実装に使う要点）
- AlphaZero系の基本は **PV-MCTS**（方策 `P` + 価値 `V` を使う探索）で、学習ターゲットは訪問回数分布 `pi` と最終勝敗 `z`。
- 自己対局時は、序盤は温度あり（探索的）、中終盤は温度を下げて確定的にする運用が一般的。
- ルートノードでは Dirichlet ノイズを加えて多様性を確保する設計が広く使われる。
- 実装面では「探索」と「NN推論（バッチ処理）」を分離するとスループットが伸びる。
- データはリプレイバッファ（スライディングウィンドウ）で管理し、古いデータを順次廃棄するのが標準的。

## 参考にした主な情報源
- [python-dlshogi: parallel_mcts_player.py](https://github.com/TadaoYamaoka/python-dlshogi/blob/master/pydlshogi/player/parallel_mcts_player.py)
- [dlshogi-zero: mcts_player.py](https://github.com/TadaoYamaoka/dlshogi-zero/blob/master/dlshogi_zero/player/mcts_player.py)
- [AlphaZero.jl Training Parameters](https://jonathan-laurent.github.io/AlphaZero.jl/stable/reference/params/)
- [AlphaZero 解説（日本語）](https://note.com/shima_7_7/n/nac459f514371)

---

## 現リポジトリ前提の設計方針

### 1) 追加するクラス/ファイル
- 新規: `explore_experiment/player/selfplay_player.py`
  - `BasePlayer` を継承
  - 既存の `MCTSPlayer` / `NPLSPlayer` と同じUSIライフサイクルを持つ
- 新規: `explore_experiment/selfplay/` 配下にデータ保存ユーティリティ
  - 例: `writer.py`, `record.py`

### 2) まずは「自己対局専用モード」を明示
- 通常対局向け挙動と混ぜると管理が難しいため、自己対局専用オプションを追加する:
  - `SelfplayMode` (bool)
  - `SelfplayTemperatureMoves` (序盤の温度適用手数、例: 20-40)
  - `SelfplayTemperature` (例: 1.0)
  - `SelfplayDirichletAlpha`, `SelfplayDirichletEpsilon`
  - `SelfplayOutputDir`

### 3) 対局中に記録する内容
- 各手で以下を記録:
  - 局面特徴量 `x`（既存 `shogi.feature` 系を流用）
  - 合法手に対する訪問回数 `N(s,a)`（最終的に正規化して `pi`）
  - 手番情報（後段で value 符号を合わせるため）
- 対局終了時に最終結果 `z in {1, 0, -1}` を全手へ逆伝播して確定

### 4) 着手選択ポリシー
- 序盤 (`move_number <= SelfplayTemperatureMoves`) は `pi` サンプリングで着手
- 以降は `argmax(N)` で着手
- これによりデータ多様性と終盤品質のバランスを取る

### 5) ルートノイズの適用
- 自己対局時のみ、ルートの事前確率 `P` に Dirichlet ノイズを注入:
  - `P' = (1-eps) * P + eps * Dir(alpha)`
- 既存対局モードでは適用しない（強さ評価の純度を保つ）

### 6) データ保存形式（最初はシンプル）
- 初期実装では `npz`/`pt` など Python から扱いやすい形式を優先
- 1レコード例:
  - `features`: `float32`
  - `policy_index`: 着手ラベル配列（可変長）
  - `policy_prob`: 上記ラベルの確率（可変長）
  - `value`: `float32`（-1/0/1）
- 将来的に `HcpeDataLoader` 互換形式へ変換するスクリプトを用意

### 7) 並列化の段階的導入
- Phase 1: 単一プロセス/単一対局ループで正しさ優先
- Phase 2: 複数自己対局ワーカー + 推論バッチ共有
- Phase 3: 生成・学習の非同期化（リプレイバッファを介した常時学習）

---

## 実装ステップ（推奨）
1. `selfplay_player.py` の雛形を作り、`go` で1局面分の `pi` を取得できるようにする。
2. 対局終了までのループ（自己対局）と棋譜/学習データ記録を実装する。
3. 温度サンプリングとルートノイズをオプション化して導入する。
4. 保存データから学習可能な最小ローダーを作る。
5. 生成速度計測（NPS, positions/sec）を入れ、ボトルネックを確認する。

## 受け入れ基準（最低限）
- 1対局が最後まで自動進行し、データファイルが生成される。
- 生成データを読み込んで、`policy loss` / `value loss` の1step学習が通る。
- 同条件で再実行した際に、データ件数・value分布が極端に崩れない。

## リスクと対策
- **探索バグで `pi` が壊れる**: ルート訪問回数合計と確率和(=1)を毎手検証してログ出力。
- **同型局面偏り**: 千日手/早期終局の統計を取り、必要なら初期局面シャッフル導入。
- **学習形式の不整合**: 早い段階で「保存形式 -> 学習入力」の変換テストを自動化。

## 次アクション
- 本メモをベースに、まず `selfplay_player.py` の最小実装（単一対局・保存あり）から着手する。
