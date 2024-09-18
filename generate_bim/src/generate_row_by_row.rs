use rand::Rng;

fn is_linearly_independent(rows: &Vec<u64>, new_row: u64, n: usize) -> bool {
    let mut row = new_row;
    for &prev_row in rows {
        let pivot_mask = prev_row.leading_zeros() as usize;
        if pivot_mask >= n {
            continue;
        }
        let pivot = 1u64 << (n - 1 - pivot_mask);
        if row & pivot != 0 {
            row ^= prev_row;
        }
    }
    row != 0
}

pub fn generate_invertible_matrix_gf2(n: usize) -> (Vec<u64>, usize) {
    let mut rng = rand::thread_rng();
    let mut rows: Vec<u64> = Vec::with_capacity(n);
    let mut tries = 0;

    while rows.len() < n {
        tries += 1;
        let new_row = if n == 64 {
            rng.gen::<u64>()
        } else {
            rng.gen_range(0..(1u64 << n))
        };

        if is_linearly_independent(&rows, new_row, n) {
            rows.push(new_row);
        }
    }

    (rows, tries)
}
