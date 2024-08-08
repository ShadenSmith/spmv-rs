
import spmv_rs

import pytest
import time
import scipy
import numpy


@pytest.mark.parametrize("rows", [1000, 10000, 20_000])
def test_spmv(rows):
    cols = rows

    S = scipy.sparse.random(rows, cols, density=0.25).tocsr()
    x = numpy.ones(cols)

    start = time.perf_counter()
    gold_b = S @ x
    gold_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    ours_b = spmv_rs.spmv(S.indptr, S.indices, S.data, x, which="basic")
    basic_elapsed = time.perf_counter() - start
    assert numpy.allclose(gold_b, ours_b)

    start = time.perf_counter()
    ours_b = spmv_rs.spmv(S.indptr, S.indices, S.data, x, which="iter")
    iter_elapsed = time.perf_counter() - start
    assert numpy.allclose(gold_b, ours_b)


    speedup = gold_elapsed / iter_elapsed
    print()
    print(f"{rows=:,d} {cols=:,d} nnz={S.nnz:,d}")
    print(f"scipy: {gold_elapsed:0.4f}s  v0: {basic_elapsed:0.4f}s v1: {iter_elapsed:0.4f}s {speedup=:0.2f}x")
