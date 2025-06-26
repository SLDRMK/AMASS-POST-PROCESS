import joblib

pkl_path = "smpl_retarget/retargeted_motion_data/mink/hmr4d_results.pkl"
data = joblib.load(pkl_path)
data = data[list(data.keys())[0]]
print(data.keys())
