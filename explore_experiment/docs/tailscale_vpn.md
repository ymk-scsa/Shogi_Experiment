# Tailscale VPNの利用方法
https://login.tailscale.com/

## 1. Tailscaleのアカウントを作成
1つアカウントを作成して必要情報を入力

## 2. Tailscaleのクライアントをインストール
Windows, Mac, Linux, iOS, AndroidなどのOSに合わせてクライアントをインストール

## 3. アカウントにデバイス登録
インストールしたのを起動してブラウザに飛びログイン。
ログインすれば自動でデバイスは登録される。
2, 3を通信したい両方のデバイスで行う。

## 4. 通信の確認
https://login.tailscale.com/admin/machines
にてデバイスのIPを確認し、ping <IP> で通信できればOK

## 5. 将棋通信確認
サーバー側で以下を実行
(WSL等の仮想環境だとオープンに受付がされないのでpowershellで実行)
```
remote/remote_engine_server.py --host 0.0.0.0 --port 49001 --engine-command ".\script\MCTS_ResNet.bat" --token Nigohachi257 --trace-io
```

クライアント側で以下を実行
```
python remote/remote_engine_client.py --host <IP> --port 49001 --token Nigohachi257
```

通信できればOK

## トラブルシューティング

### 1. `ping` は通るのに `python socket` が失敗する
- `ping` はL3疎通のみ。USI中継はTCPポート疎通が必要。
- まずクライアント側で以下を実行してTCP到達を確認する。
```
python -c "import socket; s=socket.create_connection(('<tailscale-ip>',49001),5); print('ok')"
```

### 2. `ConnectionRefusedError` が出る
- 経路は届いているが、相手がそのIP/ポートで待受していない。
- サーバー側で `netstat -ano | findstr :49001` を確認し、`LISTENING` を見る。
- `127.0.0.1:49001` しか出ない場合は外部から接続不可。`0.0.0.0:49001` で待受が必要。

### 3. `timeout` になる
- 到達経路かFWで落ちている可能性が高い。
- ただし今回多かったのは **WSL起動問題**:
  - TailscaleはWindows側、サーバーはWSL側で起動していると失敗しやすい。
  - サーバーはPowerShell側で起動する。

### 4. WSLで起動してしまっている
- プロンプトが `/mnt/c/...` ならWSLの可能性が高い。
- この場合 `--host 0.0.0.0` でも Windows の Tailscale IF で待受しない。
- 対処: PowerShellで起動する。
```
python remote/remote_engine_server.py --host 0.0.0.0 --port 49001 --engine-command ".\script\MCTS_ResNet.bat" --token <token> --trace-io
```

### 5. サーバーは起動しているのにGUIでつながらない
- `usi_proxy.py` の `--host` がTailscale IPになっているか確認（`192.168.x.x` では不可）。
- `--token` の一致を確認。
- `--trace-io` を有効にして送受信ログを比較する。

### 6. `go` の後に `stop: no active go` が出る
- `bestmove` 返却後にGUIが `stop` を送ると出ることがある。
- 通信エラーではなく、通常挙動の範囲。

### 7. `mate_move` で AssertionError が出る
- 詰み探索深さがcshogiの前提と合わない場合に起きる。
- `mate_depth` はUSI設定時に正規化されるため、最新コードへ更新する。

### 8. Ctrl+C でサーバーが止まらない
- `remote_engine_server.py` の最新版を使用する（終了処理強化済み）。
- それでも残る場合は `.bat` 側の子プロセスが残留している可能性があるため、プロセス確認して手動終了する。

### 9. 最小確認手順（切り分け順）
1. サーバー側: `netstat -ano | findstr :49001` で `0.0.0.0:49001 LISTENING` を確認
2. サーバー側: `python -c "import socket; s=socket.create_connection(('127.0.0.1',49001),5); print('ok')"`
3. サーバー側: `python -c "import socket; s=socket.create_connection(('<tailscale-ip>',49001),5); print('ok')"`
4. クライアント側: 同様に `<tailscale-ip>` へTCP接続テスト
5. 最後に `usi_proxy.py` 経由で `usi -> isready -> go` を確認
