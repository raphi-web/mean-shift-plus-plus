My implementation of MeanShift++ from the paper [Extremely Fast Mode-Seeking](https://arxiv.org/abs/2104.00303) and a spatial Meanshift akin to [OpenCV's pyrMeanShiftFiltering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9fabdce9543bd602445f5db3827e4cc0)  
It is written in Rust with Pyo3 and Rayon for parallel processing. MeanShift++ is fast, the spatial meanshift is far slower in comparison but is useful for image processing.  
Both are slow compared to GPU based alternatives.  


Wouldn't recomend both of them in production though.  

This is the spatial Mean-Shift  
![Spatial Mean Shift](https://github.com/raphi-web/mean-shift-plus-plus/blob/master/output_files/Mean-Shift-Spatial.jpg?raw=true)


This is the spatial Mean-Shift Plus Plus  
![Spatial Mean Shift Plus Plus](https://github.com/raphi-web/mean-shift-plus-plus/blob/master/output_files/Mean-Shift-pp-Spatial.jpg?raw=true)

This is the Mean-Shift Plus Plus that only works in color-/feature-space
![Spatial Mean Shift Plus Plus](https://github.com/raphi-web/mean-shift-plus-plus/blob/master/output_files/Mean-Shift_pp.jpg?raw=true)
