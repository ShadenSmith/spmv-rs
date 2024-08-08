use numpy::{
    PyArray1, PyArrayMethods, PyReadonlyArray1,
};
use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

use std::iter::zip;


fn spmv_basic(rowptr: &[i32], indices: &[i32], values: &[f64], vector: &[f64], result: &mut [f64]) {
    let num_rows = rowptr.len() - 1;
    for row_id in 0..num_rows {
        let mut accum = 0.;

        for nnz_id in rowptr[row_id]..rowptr[row_id + 1] {
            let col_id = indices[nnz_id as usize];
            let nz_val = values[nnz_id as usize];
            accum += vector[col_id as usize] * nz_val;
        }

        result[row_id] = accum;
    }
}

fn spmv_iter(rowptr: &[i32], indices: &[i32], values: &[f64], vector: &[f64], result: &mut [f64]) {
    result.par_iter_mut().enumerate().for_each(
        |(row_id, out_val)|{

        let mut accum = 0.;

        let nnz_start = rowptr[row_id] as usize;
        let nnz_stop = rowptr[row_id+1] as usize;

        let row_inds = &indices[nnz_start..nnz_stop];
        let row_vals = &values[nnz_start..nnz_stop];

        for (col_id, nz_val) in zip(row_inds.iter(), row_vals.iter()) {
            accum += vector[*col_id as usize] * nz_val;
        }

        *out_val = accum;
    });
}


#[pyfunction]
fn spmv<'py>(
    py: Python<'py>,
    csr_rowptr: PyReadonlyArray1<'py, i32>,
    csr_indices: PyReadonlyArray1<'py, i32>,
    csr_values: PyReadonlyArray1<'py, f64>,
    vector: PyReadonlyArray1<'py, f64>,
    which: String,
) -> Bound<'py, PyArray1<f64>> {
    let rowptr_array = csr_rowptr.as_array();
    let indices_array = csr_indices.as_array();
    let values_array = csr_values.as_array();
    let vec_array = vector.as_array();

    let rowptr = rowptr_array.as_slice().unwrap();
    let indices = indices_array.as_slice().unwrap();
    let values = values_array.as_slice().unwrap();
    let vec = vec_array.as_slice().unwrap();

    let num_rows = rowptr.len() - 1;
    let result = PyArray1::<f64>::zeros_bound(py, num_rows, true);

    // SAFETY: result is allocated by us and known to be 1D contiguous
    let result_vec = unsafe { result.as_slice_mut().unwrap() };

    match which.as_str() {
        "basic" => spmv_basic(rowptr, indices, values, vec, result_vec),
        "iter" => spmv_iter(rowptr, indices, values, vec, result_vec),
        _ => panic!("Unrecognized algo")
    };

    result
}

#[pymodule]
fn spmv_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spmv, m)?)?;
    Ok(())
}
