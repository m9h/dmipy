
try:
    from dipy.data import fetch_ixi_1_shot, read_ixi_1_shot
    print("IXI fetcher found: fetch_ixi_1_shot")
except ImportError:
    print("IXI fetcher fetch_ixi_1_shot NOT found")

try:
    from dipy.data import fetch_ixi_dataset
    print("IXI fetcher found: fetch_ixi_dataset")
except ImportError:
    print("IXI fetcher fetch_ixi_dataset NOT found")
