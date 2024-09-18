use rand::Rng;

fn determinant_gf2(matrix: &Vec<u64>, n: usize) -> u8 {
    let mut det: u8 = 1;

    // Create a working copy of the matrix to perform row operations
    let mut mat = matrix.clone();

    for i in 0..n {
        let mask = 1u64 << (n - 1 - i); // Mask for the i-th pivot column
        if mat[i] & mask == 0 {
            // Find a row with a non-zero element in the current column and swap
            let mut found = false;
            for j in i + 1..n {
                if mat[j] & mask != 0 {
                    mat.swap(i, j);
                    det ^= 1; // Swapping rows changes the sign of the determinant
                    found = true;
                    break;
                }
            }
            if !found {
                // If no such row exists, the matrix is singular
                return 0;
            }
        }

        // Perform row elimination to create an upper triangular matrix
        for j in i + 1..n {
            if mat[j] & mask != 0 {
                mat[j] ^= mat[i]; // XOR rows to zero out the current column in row j
            }
        }
    }

    det
}

fn generate_random_matrix_gf2(n: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    let mut matrix: Vec<u64> = vec![0; n];

    for i in 0..n {
        if n == 64 {
            // Special case for 64x64 matrix to avoid overflow
            matrix[i] = rng.gen::<u64>();
        } else {
            matrix[i] = rng.gen_range(0..(1u64 << n)); // Generate a random u64 within the range that fits the matrix size
        }
    }

    matrix
}

pub fn generate_invertible_matrix_gf2(n: usize) -> (Vec<u64>, usize) {
    let mut tries = 0;

    loop {
        tries += 1;
        let matrix = generate_random_matrix_gf2(n);
        if determinant_gf2(&matrix, n) == 1 {
            return (matrix, tries);
        }
    }
}
