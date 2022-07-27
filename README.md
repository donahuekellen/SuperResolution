# SUPER-RESOLUTION IMAGING OF REMOTE SENSED BRIGHTNESS TEMPERATURE USING A CONVOLUTIONAL NEURAL NETWORK

This is a repository of some of the work done for my master's thesis on the super-resolutioning of brightness temperature data. The full thesis can be found
here https://scholarworks.umt.edu/etd/11847/. 

The aim of the project was to extend the range of high resolution brightness temperature data available for research since
the higher resolution products only extend back to 2012. Much of the work involved identifying and testing potential input features as well as different model 
architectures.

I created and tested a few different model architectures including variations of Unets, LSTMs, RESNETs and attention
networks. I trained two separate models for the jobs of super-resolutioning 25km ssmi brightness temperature data and 25km AMSR2 brightness temperature data using a
fixed size approach as well as a third model using a tiled approach.

<p align="center"><img src="https://user-images.githubusercontent.com/37458397/181386307-3888b827-5eac-4b7a-b77e-fb01d3e324eb.png" width ="900" height="750"></p>
 <em>Comparison of LR SSMI, SR SSMI, and HR AMSR 37 GHz vertical polarization data for 2013-01-01. Shown are: (A) Norwegian Peninsula; (B) Indochinese Peninsula;
 (C) Region around the Rio De La Plata.</em>
 
 <p align="center"><img src="https://user-images.githubusercontent.com/37458397/181388393-7c49fae9-c0c1-4ce5-85bf-1e8b5933cdcd.png" width ="900" height="750"></p>
<em>Comparison of SR Tile, SR AMSR, and HR AMSR 37 GHz vertical polarization data for 2014-01-04. Shown are: (A) Norwegian Peninsula; (B) Indochinese Peninsula;
(C) Region around the Rio De La Plata.</em>
<br/><br/>
We compared our product’s performance against two
different SR techniques. The first is simple bicubic interpolation for a
baseline as is common for SR. The second we called Mean Bicubic Interpolation(MBI) which is a modified bicubic interpolation that causes less smoothing on the output.
The table below contains our results. The metrics are Mean Absolute Error(MAE), Mean Percent Error(MPE), Structural Similarity Index Measure(SSIM),
and Peak Signal-to-Noise Ratio(PSNR).

![msedge_aja7znUh7v](https://user-images.githubusercontent.com/37458397/181388674-cdc7f1b2-0bb0-4788-bf48-f88f28e9b12d.png)

<br/>
We also can look at the accuracy heatmap of the models. These help identify what regions of the earth the network struggles with. We can see that the largest errors
on shorelines and over the ocean. Water heavily affects brightness temperature readings so we expect performance to be different in pixels that contain large amounts
of water. Our focus was on land accuracies which we can see are generally within 4°K.

![srssmirep](https://user-images.githubusercontent.com/37458397/181390339-8d77ce42-5221-4369-b8fb-74ff4e4f27f4.png)
