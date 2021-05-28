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
    """The function to execute in FHE, on the untrusted server"""
    return numpy.mean(weigths)


def compile_function(function, min_weight, max_weight, num_weights):
    """Compile once for all the function"""
    config = CompilationConfig(parameter_optimizer="handselected")

    fhe_function = hnp.compile_fhe(
        function,
        {
            "weigths": hnp.encrypted_ndarray(
                bounds=(min_weight, max_weight),
                shape=(num_weights,),
            ),
        },
        config=config,
    )

    return fhe_function


def main():

    # Switch off logging / May be removed if you want to have all information
    show_logging_fhe = True

    if not show_logging_fhe:

        # Remove the default logging system
        logger.remove(0)

        # And replace it by your own, if you want
        logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

    # Settings
    min_weight = 20
    max_weight = 200
    num_weights = 3
    function = get_average_of_weights
    function_string = "get_average_of_weights"

    # 0 - Compile the function. The function definition is known to the one which compiles it, so if
    # eg, it contains confidential information that the function wants to keep private, it may be done
    # on premise
    fhe_function = compile_function(function, min_weight, max_weight, num_weights)

    # 1 - This is the key generation, done by the client on its trusted
    # device, once for all
    keys = fhe_function.create_context().keygen()

    # Private key: never give it to anyone
    secret_keys = keys.secret_keys

    # Public key: can safely be given to anyone, for FHE computation
    public_keys = keys.public_keys

    # Pick an input
    weigths = numpy.random.uniform(min_weight, max_weight, (num_weights,))

    # Computing in clear, to compare with the FHE execution
    clear_result = function(weigths)

    print(f"Using input {weigths}")

    print(f"\nCalling {function_string} in clear")
    print(f"Result in clear: {clear_result}")

    # 2 - This is the encryption, done by the client on its trusted device,
    # for each new input. Remark that this function uses keys (ie, not only
    # secret_keys) because it also needs public information
    time_start = time.time()
    enc_sample = keys.encrypt(weigths)

    print(f"\nCalling {function_string} in FHE")
    print("Encrypted input", enc_sample)

    # 3 - This is the FHE execution, done on the untrusted server
    enc_result = fhe_function.run(public_keys, enc_sample)

    print("Encrypted result after FHE computation", enc_result)

    # 4 - This is decryption, done by the client on its trusted device, for
    # each new output. Remark that this function uses keys (ie, not only
    # secret_keys) because it also needs public information
    fhe_result = keys.decrypt(enc_result)[0][0]

    print("Decrypted result as computed through the FHE computation", fhe_result)

    time_end = time.time()

    print(f"FHE computation was done in {time_end - time_start:.2f} seconds")

    diff = numpy.abs(fhe_result - clear_result)
    ratio = diff / numpy.max(clear_result)

    print(
        f"Difference between computation in clear and in FHE (expected to be as small as possible): {diff}"
    )
    print(f"Ratio (expected to be as small as possible): {100 * ratio:.2f} %")


if __name__ == "__main__":
    main()
