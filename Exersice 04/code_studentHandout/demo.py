import os
from pathlib import Path

from motion_player.amc_parser import parse_amc, parse_asf
from motion_player.Viewer3D import Viewer

if __name__ == '__main__':
    root = Path(__file__).resolve().parent / "data"
    asf1_path = root / "HDM_bk.asf"
    amc1_path = root / "HDM_bk_walk.amc"
    print(f"Loading {asf1_path}")
    joints1 = parse_asf(str(asf1_path))
    print(f"Loading {amc1_path}")
    motions1 = parse_amc(str(amc1_path))
    

    asf2_path = root / 'HDM_bk.asf'
    amc2_path = root / 'HDM_bk_walk.amc'
    print(f"Loading {asf2_path}")
    joints2 = parse_asf(str(asf2_path))
    print(f"Loading {amc2_path}")
    motions2 = parse_amc(str(amc2_path))
    
    # amc1_path = os.path.join(root, 'HDM_bk_01-01_03_120.amc')
    # amc2_path = os.path.join(root, 'changed_hand_15.amc')

    print("Starting viewer...")
    # v = Viewer((joints, joints), (motions1, motions2), legends=("first", "second"))
    # v = Viewer(joints, motions1)
    v = Viewer((joints1, joints2), (motions1, motions2), legends=("first", "second"))
    v.run()