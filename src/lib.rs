#![feature(portable_simd)]
use dashmap::DashMap;
use ndarray::parallel::prelude::*;
use numpy::{
    ndarray::{s, Array1, Array2, Array3, Axis},
    PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, ToPyArray,
};
use pyo3::prelude::*;
use std::simd::num::SimdFloat;
use std::simd::Simd;
use std::{
    i64,
    sync::atomic::{AtomicUsize, Ordering},
};

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
    let get_min_max = |ws: i64, idx: i64, length: i64| -> (i64, i64) {
        let min = if ws < idx { -ws } else { -idx };
        let max = if ws + idx < length { ws } else { length - idx };
        (min, max)
    };

    let calc_norm = |a: Simd<f64, 4>, b: Simd<f64, 4>| -> f64 {
        let diff = a - b;
        (diff * diff).reduce_sum().sqrt()
    };

    let mut image_arr = image.to_owned();
    let (h, w, c) = image_arr.dim();
    let win_size_half = win_size / 2;
    let mut new_image: Array3<f64> = Array3::zeros([h, w, c]);

    for _ in 0..max_iter {
        let new_rows: Vec<(Array2<f64>, bool)> = (0..h)
            .into_par_iter()
            .map(|i| {
                let mut converged = true;
                let (min_row, max_row) = get_min_max(win_size_half, i as i64, h as i64);
                let mut current_row: Array2<f64> = Array2::zeros([w, c]);

                for j in 0..w {
                    let pixel = Simd::from_array([
                        image_arr[[i, j, 0]],
                        image_arr[[i, j, 1]],
                        image_arr[[i, j, 2]],
                        0.0,
                    ]);
                    let mut cnt = 0;
                    let mut mean = Simd::splat(0.0);
                    let (min_col, max_col) = get_min_max(win_size_half, j as i64, w as i64);

                    for di in (min_row..max_row).step_by(1) {
                        let row = (i as i64 + di) as usize;
                        for dj in (min_col..max_col).step_by(1) {
                            let col = (j as i64 + dj) as usize;

                            let neighbour = Simd::from_array([
                                image_arr[[row, col, 0]],
                                image_arr[[row, col, 1]],
                                image_arr[[row, col, 2]],
                                0.0,
                            ]);

                            let norm = calc_norm(pixel, neighbour);

                            if norm <= color_radius {
                                mean += neighbour;
                                cnt += 1;
                            }
                        }
                    }

                    if cnt > 0 {
                        mean /= Simd::splat(cnt as f64);
                        let norm = calc_norm(pixel, mean);
                        let mean_arr: [f64; 4] = mean.to_array();

                        current_row
                            .slice_mut(s![j, ..])
                            .assign(&Array1::from_vec(mean_arr[0..3].to_vec()));

                        if norm > threshold {
                            converged = false;
                        }
                    } else {
                        let pixel_arr: [f64; 4] = pixel.to_array();
                        current_row
                            .slice_mut(s![j, ..])
                            .assign(&Array1::from_vec(pixel_arr[0..3].to_vec()));
                    }
                }

                (current_row, converged)
            })
            .collect();

        let mut converged = true;
        for (idx, (arr, conv)) in new_rows.iter().enumerate() {
            new_image.slice_mut(s![idx, .., ..]).assign(&arr);
            if !conv {
                converged = false;
            }
        }
        image_arr = new_image.clone();
        if converged {
            break;
        }
    }

    image_arr
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
    Ok(())
}
