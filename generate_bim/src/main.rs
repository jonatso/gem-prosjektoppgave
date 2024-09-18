use std::env;
use std::process;
mod generate_row_by_row;
//mod generate_whole_and_check;

fn print_matrix_python_binary_row_array(matrix: &Vec<u64>) {
    print!("[");
    for row in matrix {
        print!("0b{:b},", row);
    }
    println!("]");
}

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <matrix_size>", args[0]);
        process::exit(1);
    }

    // Parse the matrix size from the argument
    let n: usize = match args[1].parse() {
        Ok(size) => size,
        Err(_) => {
            eprintln!("Error: matrix size must be a positive integer.");
            process::exit(1);
        }
    };

    if n > 64 {
        eprintln!("Error: matrix size cannot be larger than 64.");
        process::exit(1);
    }

    let (invertible_matrix, tries) = generate_row_by_row::generate_invertible_matrix_gf2(n);
    print_matrix_python_binary_row_array(&invertible_matrix);
    println!(
        "Found an invertible matrix {}x{} over GF(2) after {} tries:",
        n, n, tries
    );
}
