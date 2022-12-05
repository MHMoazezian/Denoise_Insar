# Denoise_Insar
Noise simulation with Sentinel-1 Interferogram and using RatUNET Network to Denoise Phase and Amplitude\
1th : Create Interferigrams from Sentinel-1 data with snappy ( SNAP python GPT)\
2th : Create Wrapped Phase from SRTM(30m) DEM using Interferigrams metadata\
3th : Simulate Random_Imaginary_Gausiian noise on Wrapped Phase using Coherency\
4th : RatUNET for denoising Phase and Amplitude(Trained with almost 28000 noisy patches)\
![x9](https://user-images.githubusercontent.com/43873834/205480006-b4063e5f-d039-498b-8011-ebfbfcd07b27.png)![y9](https://user-images.githubusercontent.com/43873834/205480012-9fb51b2e-7e2c-4f2c-93cf-266fafbffc70.png)\
```docker run -it --gpus all --rm --name 'ratunet' -v $(pwd):/workspace -v /media/data/active/Remotesensing/Patches:/workspace/data ratunet```
