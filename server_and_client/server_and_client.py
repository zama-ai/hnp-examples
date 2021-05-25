import hnumpy as hnp
from hnumpy.config import CompilationConfig
import numpy
import logging
import sys
import time
from loguru import logger


def get_average_of_weights(
    weigths: numpy.ndarray,
):

    return numpy.mean(weigths)


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

    function, function_string, num_weights = (get_average_of_weights, "average weight", 5)

    array_shape = (num_weights,)

    weigths = numpy.random.uniform(min_weight, max_weight, array_shape)

    clear_result = function(weigths)

    print(f"Calling {function_string}")
    print(f"Input {weigths}")
    print(f"Clear: {clear_result}")

    # 0 - This is the compilation, done once for all
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

    # 1 - This is the key generation, done by the client on its trusted
    # device, once for all
    context = fhe_function.create_context()
    keys = context.keygen()

    # Private key: never give it to anyone
    secret_keys = keys.secret_keys

    # Public key: can safely be given to anyone, for FHE computation
    public_keys = keys.public_keys

    time_start = time.time()

    # 2 - This is the encryption, done by the client on its trusted device,
    # for each new input
    enc_sample = secret_keys.encrypt(weigths)

    # 3 - This is the FHE execution, done on the untrusted server
    enc_result = fhe_function.run(public_keys, enc_sample)

    # 4 - This is decryption, done by the client on its trusted device, for
    # each new output
    fhe_result = secret_keys.decrypt(enc_result)

    time_end = time.time()

    print(f"FHE: {function_string}: {fhe_result}, in {time_end - time_start:.2f} seconds")

    diff = numpy.abs(fhe_result - clear_result)
    ratio = diff / numpy.max(clear_result)

    print(f"Diff: {diff}")
    print(f"Ratio: {100 * ratio:.2f} %")


if __name__ == "__main__":
    main()
