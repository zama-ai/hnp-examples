import hnumpy as hnp
from hnumpy.config import CompilationConfig
import numpy


def square_perimeter(square_side_length: numpy.ndarray):
    return 4 * square_side_length


def main():
    square_lengths = numpy.arange(1, 1025, 1, dtype=numpy.float32)
    print(f"Square side lengths:\n{square_lengths}")
    clear_square_perimeters_output = square_perimeter(square_lengths)

    config = CompilationConfig(parameter_optimizer="genetic")
    # "handselected"
    # "genetic"

    fhe_square_perimeter = hnp.homomorphic_fn(
        square_perimeter,
        hnp.encrypted_ndarray(
            bounds=(square_lengths.min(), square_lengths.max()),
            shape=square_lengths.shape,
        ),
        config=config,
    )

    fhe_square_perimeter_output = fhe_square_perimeter(square_lengths)[0]

    square_perimeter_output_max_diff = numpy.abs(
        clear_square_perimeters_output - fhe_square_perimeter_output
    )

    relative_error = square_perimeter_output_max_diff / clear_square_perimeters_output

    print(f"Clear square perimeters\n{clear_square_perimeters_output}")
    print(f"FHE square perimeters\n{fhe_square_perimeter_output}")

    print(f"Max abs error square perimeters:\n{square_perimeter_output_max_diff}")
    print(
        "Max/min/mean relative error:\n"
        f"{relative_error.max()} {relative_error.min()} {relative_error.mean()}"
    )


if __name__ == "__main__":
    main()
