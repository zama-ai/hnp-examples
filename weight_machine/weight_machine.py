import hnumpy as hnp
from hnumpy.config import CompilationConfig
import numpy
import logging
import sys
import time
from loguru import logger


def get_clear_result(
    weigths: numpy.ndarray,
):

    return numpy.mean(weigths)


def get_sum_of_weights(
    weigths: numpy.ndarray,
):

    return numpy.sum(weigths)


def get_min_of_weights(
    weigths: numpy.ndarray,
):

    # hnp doesn't have numpy.min() equivalent
    a, b = numpy.split(weigths, 2)

    while a.shape[0] != 1:
        a = numpy.minimum(a, b)
        a, b = numpy.split(a, 2)

    return numpy.minimum(a, b)[0]


def get_max_of_weights(
    weigths: numpy.ndarray,
):

    # hnp doesn't have numpy.max() equivalent
    a, b = numpy.split(weigths, 2)

    while a.shape[0] != 1:
        a = numpy.maximum(a, b)
        a, b = numpy.split(a, 2)

    return numpy.maximum(a, b)[0]


list_of_functions = [
    (get_clear_result, "average weight", 100),
    (get_sum_of_weights, "sum of weights", 100),
    (get_min_of_weights, "min of weights", 4),
    (get_max_of_weights, "max of weights", 4),
]


def main():

    # Switch off logging / May be removed if you want to have all information
    show_logging_fhe = False

    if not show_logging_fhe:

        # Remove the default logging system
        logger.remove(0)

        # And replace it by your own, if you want
        logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

    # For minimum and maximum weights
    min_weight = 20
    max_weight = 200

    for function, function_string, num_weights in list_of_functions:

        print(f"\n*** Working on {function_string}: {num_weights} weights are used:\n")

        array_shape = (num_weights,)

        weigths = min_weight + (max_weight - min_weight) * numpy.random.random(array_shape).astype(
            numpy.float32
        )

        clear_result = function(weigths)

        print(f"Clear: {function_string}: {clear_result}")

        config = CompilationConfig(parameter_optimizer="genetic")

        fhe_function = hnp.compile_fhe(
            function,
            {
                "weigths": hnp.encrypted_ndarray(
                    bounds=(min_weight, max_weight),
                    shape=weigths.shape,
                ),
            },
            config=config,
        )

        context = fhe_function.create_context()
        keys = context.keygen()

        time_start = time.time()
        fhe_result = fhe_function.encrypt_and_run(keys, weigths,)[
            0
        ][0]
        time_end = time.time()

        print(f"FHE: {function_string}: {fhe_result}, in {time_end - time_start:.2f} seconds")

        diff = numpy.abs(fhe_result - clear_result)
        ratio = diff / numpy.max(clear_result)

        print(f"Diff: {diff}")
        print(f"Ratio: {100 * ratio:.2f} %")


if __name__ == "__main__":
    main()
