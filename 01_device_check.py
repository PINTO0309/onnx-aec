import sounddevice as sd

print("使用可能なデバイスの一覧:")
print(sd.query_devices())  # 使用可能なデバイスの一覧を表示

print("\nデフォルトの入力デバイス:")
print(sd.default.device[0])  # デフォルトの入力デバイスIDを表示

print("\nデフォルトの出力デバイス:")
print(sd.default.device[1])  # デフォルトの出力デバイスIDを表示






# デバイスIDを指定
input_device = 8  # Full HD webcam: USB Audio
output_device = 5  # HDA NVidia: HDMI 1

# 入力デバイスと出力デバイスの情報を取得
input_device_info = sd.query_devices(input_device, 'input')
output_device_info = sd.query_devices(output_device, 'output')

print(f"入力デバイス情報: {input_device_info}")
print(f"出力デバイス情報: {output_device_info}")

# 入力デバイスと出力デバイスのデフォルトサンプルレートを取得
input_samplerate = input_device_info['default_samplerate']
output_samplerate = output_device_info['default_samplerate']

# 共通のサンプルレートを設定（両デバイスの最小サンプルレートに合わせる）
common_samplerate = min(input_samplerate, output_samplerate)
sd.default.samplerate = common_samplerate

print(f"設定するサンプルレート: {common_samplerate} Hz")
