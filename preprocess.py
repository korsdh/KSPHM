from nptdms import TdmsFile
import pandas as pd
import os
import numpy as np
from scipy import signal, stats
import wandb

# csv보단 numpy, pte 추천

"""
| CH1 | Front Vertical Vibration |
| --- | --- |
| CH2 | Front Axial Vibration |
| CH3 | Rear Vertical Vibration |
| CH4 | Rear Axial Vibration |
| Torque[Nm] | Rotational Torque |
| TC SP Front[℃] | Front Bearing Temperature |
| TC SP Rear[℃] | Front Rear Temperature |
"""

option = False

def load_tdms_file(file_path):
    tdms_file = TdmsFile.read(file_path)

    # group_name_vibration: {CH1, CH2, CH3, CH4} 데이터, group_name_operation: {Torque[Nm], TC SP Front[℃], TC SP Rear[℃]} 데이터
    group_name_vibration = tdms_file.groups()[0].name
    group_name_operation = tdms_file.groups()[1].name
 
    vib_channels = tdms_file[group_name_vibration].channels()
    vib_data = {ch.name: ch.data for ch in vib_channels}
    
    operation_channels = tdms_file[group_name_operation].channels()
    operation_data = {ch.name: ch.data for ch in operation_channels}

    return vib_data , operation_data

# def test():
#     file_path = "Train_Set\Train1\modified_KIMM Simulator_KIMM Bearing Test_20160325122639.tdms"

#     vib_data, operation_data = load_tdms_file(file_path)

#     # 진동 데이터 출력
#     print("진동 데이터 (vibration data):")
#     for ch_name, data in vib_data.items():
#         print(f"채널명: {ch_name}, 데이터 길이: {len(data)}")
#         print(f"샘플 데이터 (앞 10개): {data[:10]}\n")

#     # operation 데이터 출력
#     print("operation 데이터 (operation data):")
#     for ch_name, data in operation_data.items():
#         print(f"채널명: {ch_name}, 데이터 길이: {len(data)}")
#         print(f"샘플 데이터 (앞 10개): {data[:10]}\n")

#     # pandas DataFrame으로 변환 (optional)
#     vib_df = pd.DataFrame({ch: vib_data[ch] for ch in vib_data})
#     operation_df = pd.DataFrame({ch: operation_data[ch] for ch in operation_data})

#     print("\n진동 데이터프레임 (앞 5행):")
#     print(vib_df[:10])

#     print("\noperation 데이터프레임 (앞 5행):")
#     print(operation_df)


# window 단위로 특징을 추출
def extract_features(window):
    return{
        'rms': np.sqrt(np.mean(window ** 2)), # Average intensity of vibration, larger value, greater vibration -> Bearing degradation, higher possibility of wear
        'kurtosis': stats.kurtosis(window), # 신호의 peakiness를 통해 급격하게 튀는 값이 있는지 알아봄
        'skewness': stats.skew(window), # 신호의 비대칭성
        'peak': np.max(np.abs(window)) # window 구간 내에서의 최대 진동값, 큰 피크 값 -> 결함, 균열
    }

def ema_smoothing(values, alpha=0.1):
    smoothed = []
    for i, v in enumerate(values):
        if i == 0:
            smoothed.append(v)
        else:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed

base_path = "Train_Set"
save_dir = "Preprocessed_CSVs"
os.makedirs(save_dir, exist_ok=True)


# hoping window (stride) - 둬 보기
# bandpass filtering
fs = 25600
lowcut = 500
highcut = 10000
b, a = signal.butter(4, [lowcut/(fs/2), highcut/(fs/2)], btype='band')

# wandb 설정
project_name = "KSPHM"
channels = ["CH1", "CH2", "CH3", "CH4"]
features = ["rms", "kurtosis", "skewness", "peak"]
ops = ["Torque[Nm]", "TC SP Front[℃]", "TC SP Rear[℃]"]

for train_folder in sorted(os.listdir(base_path)):
    train_folder_path = os.path.join(base_path, train_folder)
    if not os.path.isdir(train_folder_path):
        continue

    window_size = 6400 # 0.25 sec
    stride = 3200 # hopping 1/2

    print(f"\n▶ Starting wandb run for {train_folder}")
    run = wandb.init(project=project_name, name=f"{train_folder}_{window_size}", reinit=True)
    train_df_list = []

    for file_name in os.listdir(train_folder_path):
        if not file_name.endswith(".tdms"):
            continue

        file_path = os.path.join(train_folder_path, file_name)
        print(f"Processing file: {file_path}")

        vib_data, operation_data = load_tdms_file(file_path=file_path)

        signal_length = len(vib_data['CH1'])
        num_windows = (signal_length - window_size) // stride + 1

        # DC offset 제거 + filtering
        for ch in vib_data:
            vib_data[ch] = vib_data[ch] - np.mean(vib_data[ch])
            vib_data[ch] = signal.filtfilt(b, a, vib_data[ch])

        # num_windows = len(vib_data['CH1']) // window_size
        # windows = {ch: vib_data[ch][:num_windows*window_size].reshape(num_windows, window_size) for ch in vib_data}

        

        windows = {
            ch: np.stack([
                vib_data[ch][i*stride : i*stride + window_size]
                for i in range(num_windows)
            ])
            for ch in vib_data
        }

        output_list = [] # 각 파일마다 리스트 초기화화

        # 각 window마다 feature 추출
        for i in range(num_windows):
            row = {'train_folder': train_folder, 'file_name': file_name, 'window_id': i + 1}
            for ch in vib_data:
                feats = extract_features(windows[ch][i])
                for feat_name, feat_value in feats.items():
                    row[f'{ch}_{feat_name}'] = feat_value

            # operating data 추가
            for op_key, op_val in operation_data.items():
                row[op_key.strip()] = op_val[0]
            
            output_list.append(row)

        df_features = pd.DataFrame(output_list)
        train_df_list.append(df_features)

        csv_file_name = f"{train_folder}_{os.path.splitext(file_name)[0]}.csv"
        save_path = os.path.join(save_dir, csv_file_name)

        # CSV 저장
        df_features.to_csv(save_path, index=False)
        print(f"Saved CSV: {save_path}")

    if not train_df_list:
        continue
    train_df = pd.concat(train_df_list, ignore_index=True)

    for ch in channels:
        for feat in features:
            col = f"{ch}_{feat}"
            if col in train_df:
                data = [[i, v] for i, v in enumerate(train_df[col])]
                table = wandb.Table(data=data, columns=["step", col])
                wandb.log({f"{ch}_{feat}_trend": wandb.plot.line(table, "step", col, title=f"{ch} {feat} (Line Plot)")})

    for op in ops:
        if op in train_df:
            values = train_df[op].values
            if np.std(values) > 1e-6:  # 변화가 있는 경우만 plot
                data = [[i, v] for i, v in enumerate(values)]
                table = wandb.Table(data=data, columns=["step", op])
                wandb.log({f"{op}_trend": wandb.plot.line(table, "step", op, title=f"{op} Trend")})
            else:
                wandb.log({f"{op}_mean_value": float(np.mean(values))})

    # EMA 적용
    # for ch in channels:
    #     for feat in features:
    #         col = f"{ch}_{feat}"
    #         if col in train_df:
    #             raw_values = train_df[col].values
    #             smoothed_values = ema_smoothing(raw_values, alpha=0.1)
    #             data = [[i, v] for i, v in enumerate(smoothed_values)]
    #             table = wandb.Table(data=data, columns=["step", col])
    #             wandb.log({f"{ch}_{feat}_ema": wandb.plot.line(table, "step", col, title=f"{ch} {feat} (EMA)")})

    # for op in ops:
    #     if op in train_df:
    #         raw_values = train_df[op].values
    #         if np.std(raw_values) > 1e-6:  # 변화가 있는 경우만 plot
    #             smoothed_values = ema_smoothing(raw_values, alpha=0.1)
    #             data = [[i, v] for i, v in enumerate(smoothed_values)]
    #             table = wandb.Table(data=data, columns=["step", op])
    #             wandb.log({f"{op}_EMA": wandb.plot.line(table, "step", op, title=f"{op} (EMA)")})
    #         else:
    #             wandb.log({f"{op}_mean_value": float(np.mean(raw_values))})

    # global_step = 0
    # for ch in channels:
    #     for feat in features:
    #         col = f"{ch}_{feat}"
    #         if col in train_df:
    #             values = train_df[col].values
    #             for i, v in enumerate(values):
    #                 wandb.log({col: v}, step=global_step)

    # for op in ops:
    #     if op in train_df:
    #         values = train_df[op].values
    #         for i, v in enumerate(values):
    #             wandb.log({op: v}, step=global_step)

    run.finish()

print("전처리 완료 후 CSV 저장!")