# bin/bash

set -e  # エラー時にスクリプトを停止

# ディレクトリの用意
echo "🔧 ディレクトリの用意中..."
if [ ! -d "datas" ]; then
    mkdir datas
    cd datas
    echo "✅ ディレクトリが用意されました。"
else
    cd datas
    echo "✅ ディレクトリは既に存在します"
fi

# 年を指定 (デフォルト: 2024)
YEAR=${1:-2024}
echo "ℹ️ ${YEAR}年のデータを用意します。"

# データのダウンロード
echo "🔧 データをダウンロード中..."
if [ ! -f "wdoor${YEAR}.7z" ]; then
    wget http://wdoor.c.u-tokyo.ac.jp/shogi/x/wdoor${YEAR}.7z
    echo "✅ ダウンロードが完了しました。"
else
    echo "✅ データ(wdoor${YEAR}.7z)は既に存在します"
fi

# 7z のインストール確認
if ! command -v 7z &> /dev/null
then
    echo "7z が見つかりません。手動でインストールしてください。"
    exit 1
fi

# データを解凍
echo "🔧 データを解凍中..."
for month in {01..12}; do
    7z -aos x wdoor${YEAR}.7z ${YEAR}/*${YEAR}${month}*.csa
    echo "✅ データ(${YEAR}/*${YEAR}${month}*.csa)を解凍しました。"
done
echo "✅ データの取得が完了しました。"
