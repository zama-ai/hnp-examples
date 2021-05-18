import hnumpy as hnp
from hnumpy.config import CompilationConfig
import numpy


def square_perimeter(square_side_length: numpy.ndarray):
    return 4 * square_side_length


def square_area(square_side_length: numpy.ndarray):
    return numpy.multiply(square_side_length, square_side_length)


def run_square_perimeter():
    print("\n=================\nSquare perimeter")
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

    square_perimeter_output_abs_diff = numpy.abs(
        clear_square_perimeters_output - fhe_square_perimeter_output
    )

    relative_error = square_perimeter_output_abs_diff / clear_square_perimeters_output

    print(f"Clear square perimeters\n{clear_square_perimeters_output}")
    print(f"FHE square perimeters\n{fhe_square_perimeter_output}")

    print(f"Max abs error square perimeters:\n{square_perimeter_output_abs_diff.max()}")
    print(
        "Max/min/mean relative error:\n"
        f"{relative_error.max()} {relative_error.min()} {relative_error.mean()}"
    )


def run_square_area():
    print("\n=================\nSquare area")
    square_lengths = numpy.arange(1, 1025, 1, dtype=numpy.float32)
    print(f"Square side lengths:\n{square_lengths}")
    clear_square_areas_output = square_area(square_lengths)

    config = CompilationConfig(parameter_optimizer="genetic")
    # "handselected"
    # "genetic"

    fhe_square_area = hnp.homomorphic_fn(
        square_area,
        hnp.encrypted_ndarray(
            bounds=(square_lengths.min(), square_lengths.max()),
            shape=square_lengths.shape,
        ),
        config=config,
    )

    fhe_square_area_output = fhe_square_area(square_lengths)[0]

    square_area_output_abs_diff = numpy.abs(
        clear_square_areas_output - fhe_square_area_output
    )

    relative_error = square_area_output_abs_diff / clear_square_areas_output

    print(f"Clear square area\n{clear_square_areas_output}")
    print(f"FHE square area\n{fhe_square_area_output}")

    print(f"Max abs error square area:\n{square_area_output_abs_diff.max()}")
    print(
        "Max/min/mean relative error:\n"
        f"{relative_error.max()} {relative_error.min()} {relative_error.mean()}"
    )


def main():
    run_square_perimeter()
    run_square_area()


if __name__ == "__main__":
    main()
