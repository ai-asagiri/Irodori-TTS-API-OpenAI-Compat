# Irodori-TTS-API-OpenAI-Compat

Irodori-TTS を OpenAI 互換形式の Text-to-Speech API として利用するための FastAPI ベースの API ラッパーです。

主に `/v1/audio/speech` を提供し、OpenAI 互換 API を前提とするクライアントや UI から Irodori-TTS を利用しやすくすることを目的としています。

## 概要

このリポジトリは resident API として Irodori-TTS の runtime を保持し、OpenAI 互換形式のリクエストを Irodori-TTS の推論呼び出しに変換します。

## 本家 Irodori-TTS との関係

このリポジトリは [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) 本体ではなく、API ラッパーのみを管理します。

Irodori-TTS 本体は別ディレクトリに配置し、この API から import / runtime 経由で利用します。本家 Irodori-TTS のコードやモデル重みは、このリポジトリに同梱していません。

推論依存は本家 Irodori-TTS 側の環境に合わせます。API ラッパー側では独自に固定しません。

この API における `voice` は、OpenAI 互換 API として扱うための参照音声プリセット名です。

## 構成ファイル

* `server.py`: FastAPI API 入口
* `tts_runtime_pool.py`: resident runtime / worker pool 管理
* `config.py`: 設定読み込み
* `config.example.json`: `config.json` のテンプレート

## インストール方法

このリポジトリと本家 Irodori-TTS を同じ親ディレクトリに配置します。

```text
parent/
  Irodori-TTS/
  Irodori-TTS-API-OpenAI-Compat/
```

本家 Irodori-TTS 側の環境を作成します。

```sh
cd Irodori-TTS
uv sync
uv pip install python-dotenv
```

API ラッパー側の設定ファイルを作成します。

```sh
cd ../Irodori-TTS-API-OpenAI-Compat
cp config.example.json config.json
```

参照音声プリセットを使う場合は、API ラッパー側の `refs/` に wav を配置します。

```sh
mkdir -p refs
cp /path/to/reference.wav refs/example-voice.wav
```

## 設定

必要に応じて `config.json` を編集します。

主な設定項目:

* `irodori_root`: Irodori-TTS 本体のパス
* `output_dir`: 生成 wav の一時出力先
* `delete_delay_seconds`: FileResponse 後の一時 wav 削除待ち秒数
* `chunk_silence_seconds`: chunk 間の無音秒数
* `num_workers`: worker 数
* `audio_output_max_bytes`: 出力ディレクトリの容量上限
* `default_model`: 省略時のモデル ID
* `default_voice`: 省略時の参照音声プリセット。空文字なら参照音声なし
* `reading_replacements_path`: 読み補正 JSON のパス

## モデルと v3 runtime

この API ラッパーは Irodori-TTS v3 runtime / v3 checkpoint を既定として扱います。

* `irodori-tts`: 既定の通常モデル。Hugging Face checkpoint は `Aratako/Irodori-TTS-500M-v3` です。
* `irodori-tts-voice-design`: VoiceDesign モデル。Hugging Face checkpoint は `Aratako/Irodori-TTS-600M-v3-VoiceDesign` です。text / reference speech / caption text の 3 条件を使うため、`voice` による `refs/*.wav` 参照音声と `irodori-tts-voice-design.caption` を指定してください。

v3 では、リクエストの `common.use_duration_prediction` が既定で `true` です。この場合、各 chunk の `seconds` は runtime に `null` として渡され、v3 側の duration predictor が出力長を推定します。推定長は `common.duration_scale` でスケールできます。

従来の簡易秒数推定を使いたい場合は、Additional Parameters で `common.use_duration_prediction=false` を指定してください。その場合のみ、この API ラッパーが chunk ごとに秒数を計算して `SamplingRequest.seconds` に渡します。

OpenWebUI の Additional Parameters から API 側の文章分割を無効化する場合は、次のように指定します。

```json
{
  "common": {
    "chunking_enabled": false,
    "use_duration_prediction": true,
    "use_reading_corrections": true
  }
}
```

`common.chunking_enabled` の既定値は `true` で、従来どおり文章を chunk に分割します。`false` の場合は入力全文を 1 chunk として推論に渡します。読み補正は `common.use_reading_corrections` の設定に従って全文に適用されます。

長文を 1 回の推論に渡す場合は、モデル側の最大 text 長、VRAM 使用量、品質劣化、語尾欠けのリスクに注意してください。

## 起動方法

この API ラッパーは専用の `.venv` を持たず、Irodori-TTS 本家リポジトリの `.venv` にある Python で起動します。

このリポジトリのルートで実行します。`config.json` の `irodori_root` の既定値は `../Irodori-TTS` です。

Windows / Git Bash:

```sh
IRODORI_ROOT="$(cd ../Irodori-TTS && pwd)"
IRODORI_PYTHON="$IRODORI_ROOT/.venv/Scripts/python.exe"
"$IRODORI_PYTHON" -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Linux / macOS:

```sh
IRODORI_ROOT="$(cd ../Irodori-TTS && pwd)"
IRODORI_PYTHON="$IRODORI_ROOT/.venv/bin/python"
"$IRODORI_PYTHON" -m uvicorn server:app --host 127.0.0.1 --port 8000
```

## 依存関係

fresh `uv sync` 後、不足する API 依存は `python-dotenv` のみです。

確認:

Windows / Git Bash:

```sh
IRODORI_ROOT="$(cd ../Irodori-TTS && pwd)"
IRODORI_PYTHON="$IRODORI_ROOT/.venv/Scripts/python.exe"
"$IRODORI_PYTHON" -c 'import fastapi, uvicorn, pydantic; import dotenv; print("api deps ok")'
```

Linux / macOS:

```sh
IRODORI_ROOT="$(cd ../Irodori-TTS && pwd)"
IRODORI_PYTHON="$IRODORI_ROOT/.venv/bin/python"
"$IRODORI_PYTHON" -c 'import fastapi, uvicorn, pydantic; import dotenv; print("api deps ok")'
```

不足している場合のみ追加:

```sh
cd ../Irodori-TTS
uv pip install python-dotenv
```

本家環境と GPU/CPU 状態の確認:

Windows / Git Bash:

```sh
IRODORI_ROOT="$(cd ../Irodori-TTS && pwd)"
IRODORI_PYTHON="$IRODORI_ROOT/.venv/Scripts/python.exe"
"$IRODORI_PYTHON" -c 'import sys; print(sys.version)'
"$IRODORI_PYTHON" -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'
"$IRODORI_PYTHON" -c 'import torchaudio, transformers, accelerate, huggingface_hub; print("irodori deps ok")'
```

Linux / macOS:

```sh
IRODORI_ROOT="$(cd ../Irodori-TTS && pwd)"
IRODORI_PYTHON="$IRODORI_ROOT/.venv/bin/python"
"$IRODORI_PYTHON" -c 'import sys; print(sys.version)'
"$IRODORI_PYTHON" -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'
"$IRODORI_PYTHON" -c 'import torchaudio, transformers, accelerate, huggingface_hub; print("irodori deps ok")'
```

## 動作確認

health 確認:

```sh
curl http://127.0.0.1:8000/health
```

音声生成確認:

```sh
curl -X POST http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' -d '{"model":"irodori-tts","input":"今日はAPIの動作確認です。","response_format":"wav"}' --output speech.wav
```

## API エンドポイント

* `GET /health`: runtime / worker / 設定由来の出力先などの状態確認
* `GET /v1/models`: OpenAI 互換のモデル一覧
* `GET /v1/audio/models`: 音声モデル一覧
* `GET /v1/voices`: OpenAI 互換形式の voice 一覧
* `GET /v1/audio/voices`: 音声 API 用の voice 一覧
* `POST /v1/audio/speech`: wav 音声生成

音声生成リクエスト例:

```sh
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" -H "Content-Type: application/json" --data-raw '{"model":"irodori-tts","input":"今日は音声生成のテストです。","response_format":"wav"}' --output "speech.wav"
```

参照音声プリセットを指定する場合:

```sh
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" -H "Content-Type: application/json" --data-raw '{"model":"irodori-tts","input":"今日は音声生成のテストです。","voice":"example-voice","response_format":"wav"}' --output "speech.wav"
```

この例では、参照音声を `refs/example-voice.wav` に配置します。`voice` には、`refs/` 配下の wav ファイル名から拡張子を除いた名前を指定します。

## `voice` の扱い

`voice` は、この API で定義する参照音声プリセット名です。

* 未指定、`null`、空文字の場合: 参照音声プリセットなし
* `refs/` に配置した wav ファイルの拡張子を除いた名前: 対応する参照音声プリセットを使用
* 未知の値: 400 error

## 制限事項

* `response_format` は現状 `wav` のみ対応
* `speed` は受け取りますが、引き続き音声速度には反映されません。

## OpenAI 互換クライアント向け補足

OpenWebUI などが送信する `speed` は受け取りますが、現在は音声速度には反映されません。`1.0` 以外が指定された場合はログに記録します。

## License / Attribution

This repository is an API wrapper for Irodori-TTS.

License: MIT

Irodori-TTS is developed by Aratako and is licensed under the MIT License.

This repository does not include the original Irodori-TTS source code or model weights.

For Irodori-TTS model weights and their licensing terms, refer to the corresponding Hugging Face model cards.
