# -*- coding: utf-8 -*-

from psutil import virtual_memory
import requests
import pandas as pd
import os
import shutil

# Check available RAM
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
    print('To enable a high-RAM runtime, select the Runtime > "Change runtime type"')
    print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
    print('re-execute this cell.')
else:
    print('You are using a high-RAM runtime!\n')

# Load the data
df = pd.read_csv("/data/co_geo_coordinates.csv")
df = df.loc[300001:325000]  # 25000 images to be collected to prevent crashing ****** This is 13th batch ******

api_key = "YOUR API KEY"
url = "https://maps.googleapis.com/maps/api/staticmap?"

# Download satellite images
for i in range(25000):
    ctrl_val = df.iloc[i]['coor_cter']
    path_val = df.iloc[i]['coor_bdry']
    filename = f"/satelliteimg/{df.iloc[i]['ORIG_FID']}_{df.iloc[i]['GEOID10']}.png"
    image_url = f"{url}size=640x640&scale=2&zoom=20&center={ctrl_val}&format=png&maptype=satellite&key={api_key}"
    r = requests.get(image_url, stream=True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with a 'write binary(wb)' permission.
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        #print('Image successfully Downloaded: ', filename)
    else:
        print(f"{filename}: Image Couldn't be retrieved")

print("DONE")
