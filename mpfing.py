import os
import sys
import numpy as np
import librosa
import librosa.display
import sounddevice as sd
import wave
from scipy.ndimage import maximum_filter

###############################################################################
# 1) 基本設定
###############################################################################

# ピーク検出のパラメータ
PEAK_NEIGHBOR_SIZE = 15  # ピークを探す際の近傍範囲 (odd number: 2n+1 window)
PEAK_MIN_AMP = 10        # ピークとみなす最小振幅
FAN_VALUE = 5            # 1つのピークから最大何個の近傍ピークをファンアウトしてハッシュ化するか
HASH_TIME_DELTA_MAX = 200  # 時間差(フレーム)の上限 (あまり大きいと衝突増加)

# サンプリングレート
TARGET_SR = 16000

# 音響指紋DB（ハッシュ→ [(track_id, time_offset), ...] のリスト）
fingerprint_db = {}
# トラック名→ファイル情報のマッピング
tracks_info = {}

###############################################################################
# 2) 録音関数
###############################################################################
def record_audio(duration=10, output_path="recorded.wav", threshold=0.01):
    """
    Enterキー押下後にしきい値検知開始 → 音がthresholdを超えたらduration秒録音
    """
    samplerate = TARGET_SR
    print("Waiting for sound to start recording...")

    while True:
        buffer = sd.rec(int(0.5 * samplerate), samplerate=samplerate,
                        channels=1, dtype='float32')
        sd.wait()
        if np.max(np.abs(buffer)) > threshold:
            print("Sound detected. Recording...")
            break

    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

###############################################################################
# 3) ピーク検出
###############################################################################
def find_peaks(S, amp_min=PEAK_MIN_AMP, neighborhood=PEAK_NEIGHBOR_SIZE):
    """
    スペクトログラム S (周波数×フレーム) から局所的なピークを検出する。
    戻り値: list of (freq_idx, time_idx)
    """
    # maximum_filterで局所最大値をとる
    # (シンプルな実装であり、精度を高めるにはさらに工夫が必要)
    footprint = np.ones((neighborhood, neighborhood), dtype=bool)
    local_max = maximum_filter(S, footprint=footprint)  # 近傍での最大値
    peaks = (S == local_max)  # 元のSと等しいとこだけ True

    # 振幅が一定以上のものだけ残す
    peaks = peaks & (S >= amp_min)

    freq_idx, time_idx = np.where(peaks)
    peaks_list = list(zip(freq_idx, time_idx))
    return peaks_list

###############################################################################
# 4) ハッシュ生成 (Shazam風)
###############################################################################
def generate_hashes(peaks, fan_value=FAN_VALUE):
    """
    ピークリスト [(f, t), ...] からハッシュを生成する。
    hash = (f1, f2, delta_t) のような形 + 発生時刻 t1
    戻り値: list of (hash_str, t1)
    """
    # 時間でソート
    peaks_sorted = sorted(peaks, key=lambda x: x[1])
    result = []
    for i in range(len(peaks_sorted)):
        f1, t1 = peaks_sorted[i]
        # 近傍のファンアウト
        for j in range(1, fan_value+1):
            if (i + j) < len(peaks_sorted):
                f2, t2 = peaks_sorted[i + j]
                if (t2 - t1) <= 0:
                    continue
                if (t2 - t1) > HASH_TIME_DELTA_MAX:
                    break
                # ハッシュ化(周波数2つ + 時間差)
                h = f"{f1}|{f2}|{(t2 - t1)}"
                result.append((h, t1))
    return result

###############################################################################
# 5) オーディオからスペクトログラムを計算し、ハッシュを得る
###############################################################################
def fingerprint_audio(audio_path):
    """
    音声ファイルを読み込み、スペクトログラムを求めてピーク検出→ハッシュ生成。
    戻り値: list of (hash_str, time_offset)
    """
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    # STFTでスペクトログラム (周波数×フレーム) を得る
    D = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024))
    # ピークを検出
    peaks = find_peaks(D, amp_min=PEAK_MIN_AMP, neighborhood=PEAK_NEIGHBOR_SIZE)
    # ハッシュを生成
    hashes = generate_hashes(peaks, fan_value=FAN_VALUE)
    return hashes

###############################################################################
# 6) DB登録
###############################################################################
def add_to_database(track_id, audio_path):
    """
    指定された音声ファイルを指紋化し、fingerprint_db に登録する。
    """
    global fingerprint_db, tracks_info
    print(f"Fingerprinting track: {track_id} ...")
    hashes = fingerprint_audio(audio_path)
    # DBへ追加
    for (h, t1) in hashes:
        if h not in fingerprint_db:
            fingerprint_db[h] = []
        fingerprint_db[h].append((track_id, t1))

    # トラック情報を保持
    tracks_info[track_id] = audio_path
    print(f"Done: {track_id}, total {len(hashes)} hashes.")

###############################################################################
# 7) 照合 (クエリ音源に対してマッチするトラックを見つける)
###############################################################################
def match_audio(query_path):
    """
    query_path の指紋を生成し、DB内のトラックとマッチングする。
    もっとも一致度が高い(ハッシュ衝突が多い)トラックIDを返す。
    一定以上のスコアがない場合、「No match found」と表示する。
    """
    MIN_SCORE_THRESHOLD = 1 # マッチとみなす最低スコア
    print(f"Fingerprinting query: {query_path}")
    query_hashes = fingerprint_audio(query_path)

    # track_id -> [(offset_db, offset_query), ...] のリストをためて
    # offset_db - offset_query が揃うものが多いほどスコアが高い = 時系列の位置が揃う
    offset_diffs_map = {}

    for (h, t_query) in query_hashes:
        if h in fingerprint_db:
            matches = fingerprint_db[h]  # [(track_id, t_db), ...]
            for (track_id, t_db) in matches:
                diff = t_db - t_query
                if track_id not in offset_diffs_map:
                    offset_diffs_map[track_id] = {}
                if diff not in offset_diffs_map[track_id]:
                    offset_diffs_map[track_id][diff] = 0
                offset_diffs_map[track_id][diff] += 1

    # 各トラックごとに「最も多くのハッシュが同じtime offsetを示す」数 = スコア
    best_score = 0
    best_track_id = None
    for track_id, diffs_dict in offset_diffs_map.items():
        local_best = max(diffs_dict.values())
        if local_best > best_score:
            best_score = local_best
            best_track_id = track_id

    # 結果表示
    print("Match result:")
    if best_score >= MIN_SCORE_THRESHOLD:
        print(f"  Best match: {best_track_id} (score={best_score})")
        return best_track_id
    else:
        print("  No match found. No track is sufficiently similar.")
        return None

###############################################################################
# メイン処理例
###############################################################################
if __name__ == "__main__":
    try:
        mp3_dir = "./mp3"
        # 1) mp3フォルダ内の .mp3 ファイルをDBに登録
        for fname in os.listdir(mp3_dir):
            if fname.lower().endswith(".mp3"):
                full_path = os.path.join(mp3_dir, fname)
                track_id = fname
                add_to_database(track_id, full_path)

        print("Database build complete.\n")

        print("Press Enter key when you are ready to start waiting for sound...")
        input()

        # 2) 録音と照合のループ
        while True:
            record_path = "recorded.wav"

            # 録音
            record_audio(duration=10, output_path=record_path, threshold=0.1)

            # 照合
            matched_id = match_audio(record_path)
            if matched_id:
                print(f"Matched track: {matched_id}, file={tracks_info[matched_id]}")
            else:
                print("No matching track found in the database.")

    except KeyboardInterrupt:
        # Ctrl+Cで中断された場合の処理
        print("\nInterrupted by user.")
    finally:
        # 録音ファイルを削除
        if os.path.exists(record_path):
            os.remove(record_path)
            print(f"{record_path} has been deleted.")

