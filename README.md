My implementation of MeanShift++ from the paper [Extremely Fast Mode-Seeking](https://arxiv.org/abs/2104.00303) and a spatial Meanshift akin to [OpenCV's pyrMeanShiftFiltering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9fabdce9543bd602445f5db3827e4cc0)  
It is written in Rust with Pyo3 and Rayon for parallel processing. MeanShift++ is fast, the spatial meanshift is far slower in comparison but is useful for image processing, it considers the pixels that are within the search window.
Both are slow compared to GPU based alternatives.


I tested the proessing time:  
Mean-Shift-Spatial: 00m:36s.514ms  
Mean-Shift++: 00:03s.420ms  
Mean-Shift++-Spatial: 00:50s.125ms  

Wouldn't recomend both of them in production though.

Orginal Image  
![Spatial Mean Shift](https://github.com/raphi-web/mean-shift-plus-plus/blob/master/input_files/test-image.jpg?raw=true)

![Spatial Mean Shift](https://github.com/raphi-web/mean-shift-plus-plus/blob/master/output_files/result-1.png?raw=true)

![Mean Shift++](https://github.com/raphi-web/mean-shift-plus-plus/blob/master/output_files/result-2.png?raw=true)
