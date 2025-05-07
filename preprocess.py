from nptdms import TdmsFile
import pandas as pd

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

file_path = "Train_Set\Train1\modified_KIMM Simulator_KIMM Bearing Test_20160325122639.tdms"

vib_data, operation_data = load_tdms_file(file_path)

# 진동 데이터 출력
print("진동 데이터 (vibration data):")
for ch_name, data in vib_data.items():
    print(f"채널명: {ch_name}, 데이터 길이: {len(data)}")
    print(f"샘플 데이터 (앞 10개): {data[:10]}\n")

# operation 데이터 출력
print("operation 데이터 (operation data):")
for ch_name, data in operation_data.items():
    print(f"채널명: {ch_name}, 데이터 길이: {len(data)}")
    print(f"샘플 데이터 (앞 10개): {data[:10]}\n")

# pandas DataFrame으로 변환 (optional)
vib_df = pd.DataFrame({ch: vib_data[ch] for ch in vib_data})
operation_df = pd.DataFrame({ch: operation_data[ch] for ch in operation_data})

print("\n진동 데이터프레임 (앞 5행):")
print(vib_df[:5])

print("\noperation 데이터프레임 (앞 5행):")
print(operation_df[:5])