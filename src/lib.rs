use core::f64;
use dashmap::DashMap;
use ndarray::{parallel::prelude::*, ArrayView1, Order};
use numpy::{
    ndarray::{concatenate, s, Array1, Array2, Array3, Axis},
    PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, ToPyArray,
};
use pyo3::prelude::*;
use rayon::iter::ParallelBridge;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn mean_shift_pp_spatial(
    image: &Array3<f64>,
    band_width: f64,
    threshold: f64,
    window_size: f64,
    max_iter: usize,
) -> Array3<f64> {
    let (h, w, d) = image.dim();
    let d = d + 2;

    // create a 2D-Array of row,col indices
    let indices: Vec<f64> = (0..h)
        .into_par_iter()
        .flat_map_iter(|i| (0..w).flat_map(move |j| vec![i as f64, j as f64]))
        .collect();

    let indices_arr = Array2::from_shape_vec((h * w, 2), indices).unwrap();

    // reshape image to number of pixels * channels
    let x = image.to_shape(((h * w, d - 2), Order::RowMajor)).unwrap();
    // add the x and y coorinates to the channel axis
    let x = concatenate(Axis(1), &[indices_arr.view(), x.view()]).unwrap();
    let mut y = x.clone();
    let mut t = 1;

    // Function to calculate the grid index for a given row
    let calc_grid_idx = |row: &ArrayView1<f64>| -> Vec<i32> {
        row.iter()
            .enumerate()
            .map(|(i, &v)| {
                if i < 2 {
                    // for the spatial dimension
                    (v / window_size).floor() as i32
                } else {
                    // for the channel dimension
                    (v / band_width).floor() as i32
                }
            })
            .collect()
    };

    loop {
        // store the count and the sum of the grid blocks
        let cnt: DashMap<Vec<i32>, AtomicUsize> = DashMap::new();
        let sum: DashMap<Vec<i32>, Array1<f64>> = DashMap::new();

        // Assign points to grid cells and update counts & sums in parallel
        y.axis_iter(Axis(0)).into_par_iter().for_each(|row| {
            let grid_idx = calc_grid_idx(&row);

            cnt.entry(grid_idx.clone())
                .or_insert_with(|| AtomicUsize::new(0))
                .fetch_add(1, Ordering::SeqCst);

            sum.entry(grid_idx.clone())
                .or_insert_with(|| Array1::zeros(d))
                .iter_mut()
                .zip(row.iter())
                .for_each(|(a, &b)| *a += b);
        });

        // Compute new positions in parallel
        let y_vec: Vec<_> = y
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                let grid_idx = calc_grid_idx(&row);

                // Generate all neighbor offsets in multi-dimensional space
                let mut neighbors = vec![grid_idx.clone()];
                for j in 0..d {
                    let mut new_neighbors = vec![];
                    for neighbor in &neighbors {
                        let mut neighbor_down = neighbor.clone();
                        let mut neighbor_up = neighbor.clone();
                        neighbor_down[j] -= 1;
                        neighbor_up[j] += 1;
                        new_neighbors.push(neighbor_down);
                        new_neighbors.push(neighbor_up);
                    }
                    neighbors.extend(new_neighbors);
                }
                // compute the new sum and count
                let mut sum_s = Array1::<f64>::zeros(d);
                let mut sum_c = 0;
                for neighbor in neighbors {
                    if let Some(count) = cnt.get(&neighbor) {
                        sum_s
                            .iter_mut()
                            .zip(sum.get(&neighbor).unwrap().iter())
                            .skip(2) // Skip first two spatial coordinates
                            .for_each(|(a, &b)| *a += b);
                        sum_c += count.load(Ordering::SeqCst);
                    }
                }
                // create new rows with updated mean
                if sum_c > 0 {
                    // Only update non-spatial coordinates
                    let mut new_row = row.to_owned();
                    new_row
                        .iter_mut()
                        .skip(2)
                        .zip((sum_s / sum_c as f64).iter().skip(2))
                        .for_each(|(a, &b)| *a = b);
                    new_row.to_vec()
                } else {
                    row.to_owned().to_vec()
                }
            })
            .collect();

        let (r, _c) = y.dim();
        let y_new: Array2<f64> =
            Array2::from_shape_vec((r, d), y_vec.into_iter().flatten().collect()).unwrap();

        // Convergence check for non-spatial coordinates
        let shift: f64 = y_new
            .axis_iter(Axis(0))
            .zip(y.axis_iter(Axis(0)))
            .par_bridge()
            .map(|(new_row, old_row)| {
                new_row
                    .iter()
                    .skip(2)
                    .zip(old_row.iter().skip(2))
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>()
            })
            .sum();

        if (shift <= threshold) || (t >= max_iter) {
            break;
        }

        y = y_new;
        t += 1;
    }

    let sliced = y.slice(s![.., 2..]);
    sliced
        .to_shape(((h, w, d - 2), Order::RowMajor))
        .unwrap()
        .to_owned()
}

fn mean_shift_pp(x: &Array2<f64>, band_width: f64, threshold: f64, max_iter: usize) -> Array2<f64> {
    let (_, d) = x.dim();
    let mut y = x.clone();
    let mut t = 1;

    loop {
        // store sum and count of grid blocks
        let cnt: DashMap<Vec<i32>, AtomicUsize> = DashMap::new();
        let sum: DashMap<Vec<i32>, Array1<f64>> = DashMap::new();

        // Assign points to grid cells and update counts & sums in parallel
        y.axis_iter(Axis(0)).into_par_iter().for_each(|row| {
            let grid_idx: Vec<i32> = row
                .iter()
                .map(|&v| (v / band_width).floor() as i32)
                .collect();
            cnt.entry(grid_idx.clone())
                .or_insert_with(|| AtomicUsize::new(0))
                .fetch_add(1, Ordering::SeqCst);

            sum.entry(grid_idx.clone())
                .or_insert_with(|| Array1::zeros(d))
                .iter_mut()
                .zip(row.iter())
                .for_each(|(a, &b)| *a += b);
        });

        // Compute new positions in parallel
        let y_vec: Vec<_> = y
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                let grid_idx: Vec<i32> = row
                    .iter()
                    .map(|&v| (v / band_width).floor() as i32)
                    .collect();

                // Generate all neighbor offsets in multi-dimensional space
                let mut neighbors = vec![grid_idx.clone()];
                for j in 0..d {
                    let mut new_neighbors = vec![];
                    for neighbor in &neighbors {
                        let mut neighbor_down = neighbor.clone();
                        let mut neighbor_up = neighbor.clone();
                        neighbor_down[j] -= 1;
                        neighbor_up[j] += 1;
                        new_neighbors.push(neighbor_down);
                        new_neighbors.push(neighbor_up);
                    }
                    neighbors.extend(new_neighbors);
                }
                // calculate new sum and count
                let mut sum_s = Array1::<f64>::zeros(d);
                let mut sum_c = 0;
                for neighbor in neighbors {
                    if let Some(count) = cnt.get(&neighbor) {
                        sum_s
                            .iter_mut()
                            .zip(sum.get(&neighbor).unwrap().iter())
                            .for_each(|(a, &b)| *a += b);
                        sum_c += count.load(Ordering::SeqCst);
                    }
                }
                if sum_c > 0 {
                    // calculate new mean
                    (sum_s / sum_c as f64).to_vec()
                } else {
                    // if no neighbours return original row
                    row.to_owned().to_vec()
                }
            })
            .collect();

        let (r, _c) = y.dim();
        let y_new: Array2<f64> =
            Array2::from_shape_vec((r, d), y_vec.into_iter().flatten().collect()).unwrap();

        // Convergence check
        let shift: f64 = y_new.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum();
        if (shift <= threshold) || (t >= max_iter) {
            break;
        }

        y = y_new;
        t += 1;
    }

    y
}

fn mean_shift_spatial(
    image: Array3<f64>,
    win_size: i64,
    color_radius: f64,
    max_iter: usize,
    threshold: f64,
) -> Array3<f64> {
    // for calculating the bounds of window
    let get_valid_range = |ws: i64, idx: i64, length: i64| -> (usize, usize) {
        (
            idx.saturating_sub(ws) as usize,
            (idx + ws).min(length - 1) as usize,
        )
    };

    let (h, w, c) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let win_size_half = win_size / 2;

    // create a flattend vec of pixels
    let mut pixels: Vec<Array1<f64>> = (0..h * w)
        .into_par_iter()
        .map(|idx| {
            let i = idx / w;
            let j = idx % w;
            image.slice(s![i, j, ..]).to_owned()
        })
        .collect();

    for _ in 0..max_iter {
        let converged = AtomicBool::new(true);
        // compute new pixels in parallel
        pixels = (0..h)
            .into_par_iter()
            .flat_map(|i| {
                // range  of rows
                let (min_row, max_row) = get_valid_range(win_size_half, i as i64, h as i64);
                (0..w)
                    .into_par_iter()
                    .map(|j| {
                        // range of columns
                        let (min_col, max_col) = get_valid_range(win_size_half, j as i64, w as i64);

                        // Current pixel
                        let pixel = &pixels[i * w + j];

                        let mut mean = Array1::zeros(c);
                        let mut cnt = 0;

                        // Iterate over the neighborhood
                        for row in min_row..max_row {
                            for col in min_col..max_col {
                                let neighbor = &pixels[row as usize * w as usize + col as usize];
                                let diff = pixel - neighbor;
                                let norm = diff.mapv(|x| x.powi(2)).sum().sqrt();

                                if norm <= color_radius {
                                    // update sum and count
                                    mean += neighbor;
                                    cnt += 1;
                                }
                            }
                        }

                        // Update the pixel
                        if cnt > 0 {
                            mean /= cnt as f64;
                        } else {
                            mean.assign(&pixel);
                        }

                        let norm = (pixel - &mean).mapv(|x| x.powi(2)).sum().sqrt();
                        if norm > threshold {
                            converged.store(false, Ordering::Relaxed);
                        }

                        mean
                    })
                    .collect::<Vec<Array1<f64>>>()
            })
            .collect();
        if converged.load(Ordering::Relaxed) {
            break;
        }
    }

    let flat_values = pixels
        .par_iter()
        .map(|pxl| pxl.to_vec())
        .flatten()
        .collect::<Vec<f64>>();

    Array3::from_shape_vec((h, w, c), flat_values).unwrap()
}

#[pyfunction(name = "mean_shift_pp_spatial")]
fn mean_shift_pp_spatial_py<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, f64>,
    win_size: f64,
    color_radius: f64,
    max_iter: usize,
    threshold: f64,
) -> Bound<'py, PyArray3<f64>> {
    let arr = image.to_owned_array();
    mean_shift_pp_spatial(&arr, color_radius, threshold, win_size, max_iter).to_pyarray(py)
}

#[pyfunction(name = "mean_shift_pp")]
fn mean_shift_plus_plus_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    band_width: f64,
    threshold: f64,
    max_iter: usize,
) -> Bound<'py, PyArray2<f64>> {
    mean_shift_pp(&data.to_owned_array(), band_width, threshold, max_iter).to_pyarray(py)
}

#[pyfunction(name = "mean_shift_spatial")]
fn mean_shift_spatial_py<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, f64>,
    win_size: i64,
    color_radius: f64,
    max_iter: usize,
    threshold: f64,
) -> Bound<'py, PyArray3<f64>> {
    let image_arr = image.to_owned_array();
    mean_shift_spatial(image_arr, win_size, color_radius, max_iter, threshold).to_pyarray(py)
}

#[pymodule]
fn mean_shift(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mean_shift_spatial_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_shift_plus_plus_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_shift_pp_spatial_py, m)?)?;
    Ok(())
}
