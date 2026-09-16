import cv2
import numpy as np
import streamlit as st
from google import genai
from PIL import Image

# 画面表示の設定
st.set_page_config(
    page_title="車両特徴抽出＆判別アプリ Pro", layout="centered"
)
st.title("🚗 車両特徴抽出 ＆ 候補判別アプリ")
st.caption(
    "カメラ画像やディスプレイ接写から特徴を抽出し、候補車種を自動判定します。"
)

# APIキーの設定（Secretsまたはコード内から取得）
api_key = st.secrets.get("GEMINI_API_KEY", "")

# 1. 画像ファイルの選択（複数画像対応）
uploaded_files = st.file_uploader(
    "PCやUSB内の画像（前部・側面・後部などの複数選択可）を選択してください",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    adjusted_images = []

    st.subheader("🛠️ 1. 画像の明暗補正（暗部立ち上げ）")
    gamma = st.slider(
        "明るさ補正（ディスプレイ接写・暗部強調）",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
    )

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
        "uint8"
    )

    # 複数画像の処理と表示
    cols = st.columns(len(uploaded_files))
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # ガンマ補正
        adjusted_img_array = cv2.LUT(img_array, table)
        adjusted_image = Image.fromarray(adjusted_img_array)
        adjusted_images.append(adjusted_image)

        with cols[idx]:
            st.image(
                adjusted_image,
                caption=f"画像 {idx+1}（補正後）",
                use_container_width=True,
            )

    # 2. 人間からの補足メモ（動画目視情報）
    st.subheader("🎥 2. 人間による補足情報（任意入力）")
    video_memo = st.text_area(
        "動画で目視確認できた情報があれば入力（AIの判定精度が上がります）",
        placeholder="例：ウインカーが流れていた / スライドドアの溝が見えた / リヤワイパーが見えなかった / デイライトがL字型に光っていた 等",
    )

    # 3. 解析実行ボタン
    st.divider()
    if st.button("🔍 特徴抽出 ＆ 候補車種を判定する", type="primary"):
        if not api_key:
            st.error(
                "Gemini APIキーが設定されていません。.streamlit/secrets.toml を確認してください。"
            )
        else:
            with st.spinner(
                "画像からパーツ特徴を抽出・データベースと照合中..."
            ):
                client = genai.Client(api_key=api_key)

                # プロンプトの組み立て（前部・側面・後部を意識した指示）
                prompt = f"""
                添付された画像（防犯カメラ画像またはディスプレイ接写）を詳細に解析してください。
                【人間の目視補足メモ】: {video_memo if video_memo else '特になし'}

                複数の画像がある場合は、すべてのカットの情報を統合して判定してください。
                以下の【フォーマット】に従って明確に出力してください。

                ---
                ### 1. 部位別・視覚的特徴の抽出
                * **前部（フロント）:** ヘッドライト/デイライトの形状・発光パターン、グリルデザイン
                * **側面（サイド）:** スライドドアレールの位置（ランプの上/下）、ピラー（窓枠）の傾斜、ドアノブ配置
                * **後部（リア）:** テールランプの発光ライン（一文字/L字/独立）、ナンバープレート位置、ルーフスポイラー/ワイパー格納の有無

                ### 2. 可能性の高い候補車種（上位3案）
                **第1候補:** 【メーカー名 車種名（推定世代/年式）】 (確信度: XX%)
                * **一致する理由:**
                * **懸念点・不一致点:**

                **第2候補:** 【メーカー名 車種名（推定世代/年式）】 (確信度: XX%)
                * **一致する理由:**
                * **懸念点・不一致点:**

                **第3候補:** 【メーカー名 車種名（推定世代/年式）】 (確信度: XX%)
                * **一致する理由:**
                * **懸念点・不一致点:**
                ---
                """

                # APIへ送信（画像リスト + プロンプト）
                contents = adjusted_images + [prompt]
                response = client.models.generate_content(
                    model="gemini-2.5-flash", contents=contents
                )

                # 結果表示
                st.success("解析が完了しました")
                st.markdown(response.text)

                # 4. 人間用ダブルチェックリスト
                st.subheader("👁️ 3. 人間による最終目視チェック項目")
                st.info(
                    "AIの誤認を防ぐため、以下のポイントを画像・動画で最終確認してください。"
                )
                st.checkbox(
                    "白飛びで独立したライトが繋がって（一文字に）見えていないか"
                )
                st.checkbox(
                    "スライドドアの溝（レール）の位置はライトの下を通っているか"
                )
                st.checkbox(
                    "リヤワイパーは露出しているか、スポイラー内に格納されているか"
                )
                st.checkbox(
                    "ディスプレイ直撮りによるアスペクト比（縦横比）の歪みはないか"
                )