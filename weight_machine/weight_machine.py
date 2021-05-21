import hnumpy as hnp
from hnumpy.config import CompilationConfig
import numpy
import logging


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
        # logging.basicConfig(level=logging.ERROR)
        logging.basicConfig(level=logging.CRITICAL)

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

        fhe_result = fhe_function.encrypt_and_run(
            keys,
            weigths,
        )[0]

        print(f"FHE: {function_string}: {fhe_result}")


if __name__ == "__main__":
    main()
