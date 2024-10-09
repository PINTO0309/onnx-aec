import numpy as np
import sounddevice as sd
import onnxruntime
import soundfile as sf
import pyaudio
import librosa  # 高精度のリサンプリングを行うために使用
from scipy.signal import resample_poly  # 高精度リサンプリングを行うための関数


class DECModel:
    def __init__(self, model_path, window_length, hop_fraction, dft_size, hidden_size, sampling_rate=16000):
        self.hop_fraction = hop_fraction
        self.dft_size = dft_size
        self.hidden_size = hidden_size
        self.sampling_rate = sampling_rate
        self.frame_size = int(window_length * sampling_rate)  # ウィンドウサイズ（フレームサイズ）を設定
        self.hop_size = int(self.frame_size * hop_fraction)  # オーバーラップサイズを設定
        self.window = np.sqrt(np.hanning(self.frame_size + 1)[:-1]).astype(np.float32)  # ハン窓を適用
        self.model = onnxruntime.InferenceSession(model_path)
        self.h01 = np.zeros((1, 1, hidden_size), dtype=np.float32)
        self.h02 = np.zeros((1, 1, hidden_size), dtype=np.float32)

    def calc_features(self, xmag_mic, xmag_far):
        feat_mic = np.log10(np.maximum(xmag_mic**2, 1e-12))
        feat_far = np.log10(np.maximum(xmag_far**2, 1e-12))
        feat = np.concatenate([feat_mic, feat_far])
        feat /= 20.  # 正規化
        return feat[np.newaxis, np.newaxis, :].astype(np.float32)

    def enhance_frame(self, mic_frame, far_frame):
        cspec_mic = np.fft.rfft(mic_frame * self.window, self.dft_size)
        xmag_mic, xphs_mic = np.abs(cspec_mic), np.angle(cspec_mic)
        cspec_far = np.fft.rfft(far_frame * self.window, self.dft_size)
        xmag_far = np.abs(cspec_far)
        feat = self.calc_features(xmag_mic, xmag_far)
        inputs = {"input": feat, "h01": self.h01, "h02": self.h02}
        mask, self.h01, self.h02 = self.model.run(None, inputs)
        enhanced_frame = np.fft.irfft(mask[0, 0] * xmag_mic * np.exp(1j * xphs_mic), self.dft_size) * self.window
        return enhanced_frame[:self.frame_size]


class RealTimeEchoCanceller:
    def __init__(self, model, sampling_rate=16000, frame_size=320, output_file="output.wav"):
        self.model = model
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.buffer_mic = np.zeros(frame_size)
        self.buffer_far = np.zeros(frame_size)
        self.enhanced_audio = []  # エコーキャンセルされた音声を保存するリスト
        self.output_file = output_file  # 出力ファイル名

        # PyAudio の設定
        self.pyaudio_instance = pyaudio.PyAudio()
        self.far_end_stream = self.pyaudio_instance.open(format=pyaudio.paFloat32,
                                                         channels=1,  # システムの出力はモノラルで取得
                                                         rate=self.sampling_rate,
                                                         input=True,
                                                         frames_per_buffer=self.frame_size,
                                                         input_device_index=None)

    def get_far_end_audio(self):
        """システムの出力音を取得して遠端音とする"""
        far_end_data = self.far_end_stream.read(self.frame_size, exception_on_overflow=False)
        return np.frombuffer(far_end_data, dtype=np.float32)

    def audio_callback(self, indata, outdata, frames, time, status):
        """リアルタイムでエコーキャンセルを行い、エコーキャンセルされた音声を保存する"""
        if status:
            print(status)

        # マイク入力を取得（近端音）
        mic_input = indata[:, 0]  # 1チャンネルのマイク入力を取得

        # 遠端音（ファーエンド）を取得
        far_input = self.get_far_end_audio()

        # デバイスのサンプルレートとモデルのサンプルレートが異なる場合は、リサンプリングを行う
        if self.sampling_rate != self.model.sampling_rate:
            # リサンプリング時に高精度な `librosa.resample` を使用
            mic_input_downsampled = librosa.resample(mic_input, orig_sr=self.sampling_rate, target_sr=self.model.sampling_rate)
            far_input_downsampled = librosa.resample(far_input, orig_sr=self.sampling_rate, target_sr=self.model.sampling_rate)
        else:
            mic_input_downsampled = mic_input
            far_input_downsampled = far_input

        # バッファのサイズを確認し、必要であればリサイズする
        if len(self.buffer_mic) != self.model.frame_size:
            self.buffer_mic = np.zeros(self.model.frame_size)
        if len(self.buffer_far) != self.model.frame_size:
            self.buffer_far = np.zeros(self.model.frame_size)

        # バッファにデータを追加
        self.buffer_mic = np.roll(self.buffer_mic, -self.model.frame_size)
        self.buffer_mic[-self.model.frame_size:] = mic_input_downsampled[:self.model.frame_size]  # サイズをモデルのフレームサイズに合わせる

        self.buffer_far = np.roll(self.buffer_far, -self.model.frame_size)
        self.buffer_far[-self.model.frame_size:] = far_input_downsampled[:self.model.frame_size]  # サイズをモデルのフレームサイズに合わせる

        # エコーキャンセルを行う（バッファがフレームサイズ以上になったら）
        if len(self.buffer_mic) >= self.model.frame_size and len(self.buffer_far) >= self.model.frame_size:
            mic_chunk = self.buffer_mic[-self.model.frame_size:]
            far_chunk = self.buffer_far[-self.model.frame_size:]
            enhanced_chunk = self.model.enhance_frame(mic_chunk, far_chunk)

            # スピーカー出力は通常の状態を維持し、エコーキャンセルされた音声を保存
            self.enhanced_audio.extend(enhanced_chunk)

    def start_stream(self):
        """リアルタイムエコーキャンセルのストリームを開始"""
        with sd.Stream(callback=self.audio_callback,
                       blocksize=self.frame_size,  # フレームサイズとブロックサイズを一致させる
                       channels=1,                 # チャンネル数は1（モノラル）に設定
                       samplerate=self.sampling_rate):  # デバイスのサンプルレートを指定
            print("リアルタイムエコーキャンセルを開始しています。Ctrl+Cで停止できます。")
            try:
                sd.sleep(int(60 * 1e3))  # 60秒間録音（Ctrl+Cで停止）
            except KeyboardInterrupt:
                print("録音を停止しました。")

        # エコーキャンセルされた音声をWAVファイルとして保存
        self.save_to_wav()

    def save_to_wav(self):
        """エコーキャンセルされた音声をWAVファイルに保存"""
        # 保存時にモデルのサンプルレートを指定してWAVに保存
        enhanced_audio_np = np.array(self.enhanced_audio)

        # 音声データの振動やノイズを抑えるため、音量を正規化
        enhanced_audio_np /= np.max(np.abs(enhanced_audio_np)) + 1e-7  # 安定性を考慮して 1e-7 を追加

        # `sf.write` でWAVファイルに保存
        sf.write(self.output_file, enhanced_audio_np, self.model.sampling_rate)
        print(f"エコーキャンセルされた音声を {self.output_file} に保存しました。")


if __name__ == "__main__":
    # ONNXモデルのパスを指定
    model_path = "dec-baseline-model-icassp2022.onnx"

    # DECモデルの初期化
    model = DECModel(model_path, window_length=0.02, hop_fraction=0.5, dft_size=320, hidden_size=322, sampling_rate=16000)

    # リアルタイムエコーキャンセラの初期化
    canceller = RealTimeEchoCanceller(model, sampling_rate=16000, frame_size=model.frame_size, output_file="enhanced_output.wav")

    # リアルタイムエコーキャンセルのストリームを開始
    canceller.start_stream()

