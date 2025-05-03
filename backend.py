import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2 # cv2 をインポート
import sys # sys をインポート

# --- 設定 ---
WEIGHTS_PATH = 'yolov8n.pt'
CONF_THRESHOLD = 0.5
TEMP_UPLOAD_DIR = "temp_uploads"
TEMP_OUTPUT_DIR = "temp_outputs"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

# --- モデルのロード ---
# API起動時に一度だけモデルをロードする
try:
    model = YOLO(WEIGHTS_PATH)
    print(f"モデル '{WEIGHTS_PATH}' をロードしました。")
except Exception as e:
    print(f"致命的エラー: モデル '{WEIGHTS_PATH}' のロードに失敗しました。APIを起動できません。", file=sys.stderr)
    print(e, file=sys.stderr)
    # ここで終了するか、エラー状態を示すフラグを設定するなどの処理が必要
    # 簡単のため、ここではエラーを出力して続行するが、実際には起動失敗とすべき
    model = None # モデルがロードできなかったことを示す

app = FastAPI()

@app.on_event("shutdown")
def cleanup_temp_dirs():
    """APIシャットダウン時に一時ディレクトリをクリーンアップ"""
    print("一時ディレクトリをクリーンアップします...")
    try:
        if os.path.exists(TEMP_UPLOAD_DIR):
            shutil.rmtree(TEMP_UPLOAD_DIR)
            print(f"ディレクトリ '{TEMP_UPLOAD_DIR}' を削除しました。")
        if os.path.exists(TEMP_OUTPUT_DIR):
            shutil.rmtree(TEMP_OUTPUT_DIR)
            print(f"ディレクトリ '{TEMP_OUTPUT_DIR}' を削除しました。")
    except Exception as e:
        print(f"警告: 一時ディレクトリのクリーンアップ中にエラーが発生しました: {e}", file=sys.stderr)

async def run_detection(image_path: str) -> str:
    """
    指定された画像パスに対して物体検出を実行し、結果画像のパスを返す。
    main.py の画像処理ロジックをベースにする。
    """
    print(f"run_detection: image_path={image_path}") # ログを追加
    if model is None:
        print("run_detection: model is None") # ログを追加
        raise HTTPException(status_code=500, detail="モデルがロードされていません。")

    try:
        print(f"run_detection: 入力画像 '{image_path}' に対して推論を実行します...")
        # predictメソッドはResultsオブジェクトのリストを返す
        # save=True で結果を保存するが、保存先を一時ディレクトリにする
        # project と name を指定して一時ディレクトリに保存させる
        unique_id = str(uuid.uuid4())
        output_subdir = os.path.join(TEMP_OUTPUT_DIR, unique_id) # 各リクエストにユニークなサブディレクトリを作成
        os.makedirs(output_subdir, exist_ok=True)

        results_list = model.predict(image_path, save=True, conf=CONF_THRESHOLD, project=TEMP_OUTPUT_DIR, name=unique_id, exist_ok=True)

        # 保存されたパスを取得
        saved_path = None
        if results_list and hasattr(results_list[0], 'save_dir'):
            save_dir = results_list[0].save_dir # これが output_subdir と同じはず
            input_filename = os.path.basename(image_path)
            # YOLOは入力ファイル名で保存するので、そのパスを組み立てる
            potential_saved_path = os.path.join(save_dir, input_filename)
            if os.path.exists(potential_saved_path):
                saved_path = potential_saved_path
            else:
                 # 見つからない場合、ディレクトリ内の最初の画像ファイルを探す (代替策)
                 try:
                     img_files = [f for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
                     if img_files:
                         saved_path = os.path.join(save_dir, img_files[0])
                 except Exception as e_find:
                     print(f"警告: 保存された画像ファイルの検索中にエラー: {e_find}")
                     pass # エラーでも続行

        if saved_path and os.path.exists(saved_path):
            print(f"推論が完了し、結果が '{saved_path}' に保存されました。")
            return saved_path
        else:
            print(f"エラー: 推論結果の画像ファイルが見つかりませんでした。検索ディレクトリ: {output_subdir}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="推論結果の保存に失敗しました。")

    except Exception as e:
        print(f"エラー: 画像の推論または保存中にエラーが発生しました。入力パス '{image_path}'", file=sys.stderr)
        print(e, file=sys.stderr)
        # エラーの詳細をクライアントに返しすぎないように注意
        raise HTTPException(status_code=500, detail=f"推論中に内部エラーが発生しました: {str(e)}")


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    画像ファイルをアップロードし、物体検出を実行して結果画像を返すエンドポイント。
    """
    print("predict_image: ファイルを受信") # ログを追加
    if model is None:
        print("predict_image: model is None") # ログを追加
        raise HTTPException(status_code=503, detail="モデルが利用不可です。サーバーを確認してください。")

    # ファイル拡張子のチェック (任意だが推奨)
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        print(f"predict_image: 無効なファイル形式: {file_ext}") # ログを追加
        raise HTTPException(status_code=400, detail=f"無効なファイル形式です。許可されている形式: {', '.join(allowed_extensions)}")

    # 一時ファイルとして保存
    temp_input_path = os.path.join(TEMP_UPLOAD_DIR, f"{uuid.uuid4()}{file_ext}")
    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"predict_image: アップロードされたファイルを '{temp_input_path}' に一時保存しました。")
    except Exception as e:
        print(f"predict_image: エラー: アップロードファイルの保存中にエラー: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="ファイルのアップロード処理中にエラーが発生しました。")
    finally:
        # file.file を閉じる (FastAPIが自動で閉じるかもしれないが念のため)
        await file.close()

    # 推論の実行
    try:
        result_image_path = await run_detection(temp_input_path)
    except HTTPException as e:
        # run_detection 内で発生した HTTP 例外をそのまま再送出
        # 一時入力ファイルは finally で削除される
        raise e
    except Exception as e:
        # run_detection 内で予期せぬエラーが発生した場合
        print(f"predict_image: エラー: run_detection で予期せぬエラー: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="推論処理中に予期せぬエラーが発生しました。")
    finally:
        # 入力用の一時ファイルを削除
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
                print(f"predict_image: 一時入力ファイル '{temp_input_path}' を削除しました。")
            except Exception as e_remove:
                print(f"predict_image: 警告: 一時入力ファイル '{temp_input_path}' の削除中にエラー: {e_remove}", file=sys.stderr)

    # 結果画像をレスポンスとして返す
    # FileResponse は自動的にファイルを閉じるので、後処理でファイルを削除する必要がある
    # ここでファイルを返した後、削除するコールバックを設定する
    try:
        # FileResponse に background task を設定して、レスポンス送信後にファイルを削除する
        return FileResponse(result_image_path, media_type=f"image/{file_ext.lstrip('.')}", 
                            #background=BackgroundTask(os.remove, result_image_path))
        )
    except Exception as e:
        print(f"エラー: 結果ファイルのレスポンス作成中にエラー: {e}", file=sys.stderr)
        # 結果ファイルが既に存在しない場合なども考慮
        raise HTTPException(status_code=500, detail="結果ファイルの送信中にエラーが発生しました。")

# BackgroundTask をインポート
from starlette.background import BackgroundTask

# if __name__ == "__main__":
#     import uvicorn
#     print("FastAPIサーバーを起動します (http://127.0.0.1:8000)")
#     # Uvicornサーバーを起動
#     # reload=True は開発時に便利だが、本番環境では False にする
#     uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
