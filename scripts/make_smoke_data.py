"""Generate SYNTHETIC test-only images and labels; never use as scientific results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--out',default='smoke_data');a=p.parse_args()
root=Path(a.out);(root/'images').mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(7)
source=['population_count','education_accessibility','food_accessibility']
target=['healthcare_accessibility','mobility_opportunity']
for name,cities,tasks,csp,isp in [('train',['A','B'],source,'seen_city','train_indicator'),('scui',['A'],target,'seen_city','test_indicator'),('ucsi',['C'],source,'unseen_city','train_indicator'),('ucui',['C'],target,'unseen_city','test_indicator')]:
    rows=[]
    for city in cities:
      for j in range(4):
        fn=f'{city}_{j}.png';file=root/'images'/fn
        if not file.exists():Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(file)
        for task in tasks:
            rows.append(dict(city=city,city_query=city,image_name=fn,patch_lon=10.,patch_lat=20.,patch_area_km2=4.,indicator_column=task,indicator_value_log1p=float(rng.random()),city_split=csp,indicator_split=isp))
    pd.DataFrame(rows).to_csv(root/f'{name}.csv',index=False)
print('Generated synthetic-only inputs at',root)
