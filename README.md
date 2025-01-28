My implementation of MeanShift++ from the paper ["Extremely Fast Mode-Seeking" ](https://www.google.com](https://openaccess.thecvf.com/content/CVPR2021/papers/Jang_MeanShift_ExtremelyB_Fast_Mode-Seeking_With_Applications_to_Segmentation_and_Object_CVPR_2021_paper.pdf) 
and a spatial Meanshift akin to [OpenCV's pyrMeanShiftFiltering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9fabdce9543bd602445f5db3827e4cc0)
It is written in Rust with Pyo3 and Rayon far parallel processing. MeanShift++ is fast, the spatial meanshift is far slower in comparison but is useful for image processing. 
Both are slow compared to GPU based alternatives.
Wouldn't recomend both of them in production though.



