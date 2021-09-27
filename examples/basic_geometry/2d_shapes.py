import hnumpy as hnp
from hnumpy.config import CompilationConfig
import numpy


def square_perimeter(square_side_length: numpy.ndarray):
    return 4 * square_side_length


def square_area(square_side_length: numpy.ndarray):
    # return numpy.multiply(square_side_length, square_side_length)
    return square_side_length ** 2


def rect_perimeter(rect_sides_length: numpy.ndarray):
    return 2 * (rect_sides_length[:, 0] + rect_sides_length[:, 1])


def rect_area(rect_sides_length: numpy.ndarray):
    return numpy.multiply(rect_sides_length[:, 0], rect_sides_length[:, 1])


def run_square_perimeter(square_lengths: numpy.ndarray):
    print("\n=================\nSquare perimeter")
    print(f"Square side lengths:\n{square_lengths}")
    clear_square_perimeters_output = square_perimeter(square_lengths)

    # config = CompilationConfig(parameter_optimizer="genetic", examples=(square_lengths,))
    config = CompilationConfig(parameter_optimizer="genetic")
    # "handselected"
    # "genetic"

    fhe_square_perimeter = hnp.compile_fhe(
        square_perimeter,
        {
            "square_side_length": hnp.encrypted_ndarray(
                bounds=(square_lengths.min(), square_lengths.max()),
                shape=square_lengths.shape,
            )
        },
        config=config,
    )

    context = fhe_square_perimeter.create_context()
    keys = context.keygen()

    fhe_square_perimeter_output = fhe_square_perimeter.encrypt_and_run(keys, square_lengths)
    square_perimeter_output_abs_diff = numpy.abs(
        clear_square_perimeters_output - fhe_square_perimeter_output
    )

    relative_error = square_perimeter_output_abs_diff / clear_square_perimeters_output

    print(f"Clear square perimeters\n{clear_square_perimeters_output}")
    print(f"FHE square perimeters\n{fhe_square_perimeter_output}")

    print(f"Expected precision: {fhe_square_perimeter.expected_precision()}")

    print(f"Max abs error square perimeters:\n{square_perimeter_output_abs_diff.max()}")
    print("Mean absolute error:\n" f"{square_perimeter_output_abs_diff.mean()}")
    print(
        "Max/min/mean relative error:\n"
        f"{relative_error.max()} {relative_error.min()} {relative_error.mean()}"
    )


def run_square_area(square_lengths: numpy.ndarray):
    print("\n=================\nSquare area")
    print(f"Square side lengths:\n{square_lengths}")
    clear_square_areas_output = square_area(square_lengths)

    # config = CompilationConfig(parameter_optimizer="genetic", examples=(square_lengths,))
    config = CompilationConfig(parameter_optimizer="genetic")
    # "handselected"
    # "genetic"

    fhe_square_area = hnp.compile_fhe(
        square_area,
        {
            "square_side_length": hnp.encrypted_ndarray(
                bounds=(square_lengths.min(), square_lengths.max()),
                shape=square_lengths.shape,
            )
        },
        config=config,
    )

    context = fhe_square_area.create_context()
    keys = context.keygen()

    fhe_square_area_output = fhe_square_area.encrypt_and_run(keys, square_lengths)

    square_area_output_abs_diff = numpy.abs(clear_square_areas_output - fhe_square_area_output)

    relative_error = square_area_output_abs_diff / clear_square_areas_output

    print(f"Clear square area\n{clear_square_areas_output}")
    print(f"FHE square area\n{fhe_square_area_output}")

    print(f"Expected precision: {fhe_square_area.expected_precision()}")

    print(f"Max abs error square area:\n{square_area_output_abs_diff.max()}")
    print("Mean absolute error:\n" f"{square_area_output_abs_diff.mean()}")
    print(
        "Max/min/mean relative error:\n"
        f"{relative_error.max()} {relative_error.min()} {relative_error.mean()}"
    )


def run_square():
    square_lengths = numpy.arange(1, 1025, 1, dtype=numpy.float32)
    run_square_perimeter(square_lengths)
    run_square_area(square_lengths)


def run_rect_perimeter(rect_sides_lengths: numpy.ndarray):
    print("\n=================\nRect perimeter")
    print(f"Rect sides lengths:\n{rect_sides_lengths}")
    clear_rect_perimeter_output = rect_perimeter(rect_sides_lengths)

    # config = CompilationConfig(parameter_optimizer="genetic", examples=(rect_sides_lengths,))
    config = CompilationConfig(parameter_optimizer="genetic")
    # "handselected"
    # "genetic"

    fhe_rect_perimeter = hnp.compile_fhe(
        rect_perimeter,
        {
            "rect_sides_length": hnp.encrypted_ndarray(
                bounds=(rect_sides_lengths.min(), rect_sides_lengths.max()),
                shape=rect_sides_lengths.shape,
            )
        },
        config=config,
    )

    context = fhe_rect_perimeter.create_context()
    keys = context.keygen()

    fhe_rect_perimeter_output = fhe_rect_perimeter.encrypt_and_run(keys, rect_sides_lengths)

    rect_perimeter_output_abs_diff = numpy.abs(
        clear_rect_perimeter_output - fhe_rect_perimeter_output
    )

    relative_error = rect_perimeter_output_abs_diff / clear_rect_perimeter_output

    print(f"Clear rect perimeter\n{clear_rect_perimeter_output}")
    print(f"FHE rect perimeter\n{fhe_rect_perimeter_output}")

    print(f"Expected precision: {fhe_rect_perimeter.expected_precision()}")

    print(f"Max abs error rect perimeter:\n{rect_perimeter_output_abs_diff.max()}")
    print(
        "Max/min/mean relative error:\n"
        f"{relative_error.max()} {relative_error.min()} {relative_error.mean()}"
    )


def run_rect_area(rect_sides_lengths: numpy.ndarray):
    print("\n=================\nRect area")
    print(f"Rect sides lengths:\n{rect_sides_lengths}")
    clear_rect_area_output = rect_area(rect_sides_lengths)

    # config = CompilationConfig(parameter_optimizer="genetic", examples=(rect_sides_lengths,))
    config = CompilationConfig(parameter_optimizer="genetic")
    # "handselected"
    # "genetic"

    fhe_rect_area = hnp.compile_fhe(
        rect_area,
        {
            "rect_sides_length": hnp.encrypted_ndarray(
                bounds=(rect_sides_lengths.min(), rect_sides_lengths.max()),
                shape=rect_sides_lengths.shape,
            )
        },
        config=config,
    )

    context = fhe_rect_area.create_context()
    keys = context.keygen()

    fhe_rect_area_output = fhe_rect_area.encrypt_and_run(keys, rect_sides_lengths)

    rect_area_output_abs_diff = numpy.abs(clear_rect_area_output - fhe_rect_area_output)

    relative_error = rect_area_output_abs_diff / clear_rect_area_output

    print(f"Clear rect area\n{clear_rect_area_output}")
    print(f"FHE rect area\n{fhe_rect_area_output}")

    print(f"Expected precision: {fhe_rect_area.expected_precision()}")

    print(f"Max abs error rect area:\n{rect_area_output_abs_diff.max()}")
    print(
        "Max/min/mean relative error:\n"
        f"{relative_error.max()} {relative_error.min()} {relative_error.mean()}"
    )


def run_rect():
    rect_sides_lengths = 1024.0 * numpy.random.random((1024, 2)).astype(numpy.float32)
    run_rect_perimeter(rect_sides_lengths)
    run_rect_area(rect_sides_lengths)


def main():
    run_square()
    run_rect()


if __name__ == "__main__":
    main()
