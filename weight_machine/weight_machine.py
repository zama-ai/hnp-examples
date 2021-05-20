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


list_of_functions = [
    (get_clear_result, "average weight"),
    (get_sum_of_weights, "sum of weights"),
]


def main():

    # Switch off logging / May be removed if you want to have all information
    show_logging_fhe = False

    if not show_logging_fhe:
        # logging.basicConfig(level=logging.ERROR)
        logging.basicConfig(level=logging.CRITICAL)

    # Number of weights
    num_weights = 100

    # For minimum and maximum weights
    min_weight = 20
    max_weight = 200

    array_shape = (num_weights,)

    weigths = min_weight + (max_weight - min_weight) * numpy.random.random(array_shape).astype(
        numpy.float32
    )

    for function, function_string in list_of_functions:

        print("\n*** Working on {function_string}:\n")

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
