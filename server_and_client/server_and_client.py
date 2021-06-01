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
    print(f"Compiling the function")
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


def user_generates_its_key(fhe_function):
    """Done by the user on its private and secure device"""
    print(f"Generating keys")
    keys = fhe_function.create_context().keygen()

    # Public key: can safely be given to anyone, for FHE computation
    public_keys = keys.public_keys

    return keys, public_keys


def user_picks_input_and_encrypts(
    function, function_string, keys, min_weight, max_weight, num_weights
):
    """Done by the user on its private and secure device, with its private keys"""

    # Pick an input
    weigths = numpy.random.uniform(min_weight, max_weight, (num_weights,))
    print(f"    Picking inputs {weigths}")

    encrypted_weights = keys.encrypt(weigths)

    # Also, for comparison, we compute here the expected result
    time_start = time.time()
    clear_result = function(weigths)
    time_end = time.time()

    print(f"\n    Calling {function_string} in clear")
    print(f"    Result in clear: {clear_result}")
    print(f"    Clear computation was done in {time_end - time_start:.2f} seconds")

    return encrypted_weights, clear_result


def running_fhe_computation_on_untrusted_server(
    fhe_function, function_string, public_keys, encrypted_weights
):
    """Done on the untrusted server, but still preserves the user's privacy, thanks
    to the FHE properties. Only public keys are used"""
    print(f"\n    Calling {function_string} in FHE")
    print(f"    Encrypted input shape: {encrypted_weights.shape}")

    time_start = time.time()
    encrypted_result = fhe_function.run(public_keys, encrypted_weights)
    time_end = time.time()

    print(f"    Encrypted result shape after FHE computation: {encrypted_result.shape}")
    print(f"    FHE computation was done in {time_end - time_start:.2f} seconds")

    return encrypted_result


def user_decrypts(keys, encrypted_result):
    """Done by the user on its private and secure device, with its private keys"""
    fhe_result = keys.decrypt(encrypted_result)[0]

    print(f"    Decrypted result as computed through the FHE computation: {fhe_result}")

    return fhe_result


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
    num_weights = 10
    function = get_average_of_weights
    function_string = "get_average_of_weights"

    # 0 - Compile the function. The function definition is known to the one which compiles it, so if
    # eg, it contains confidential information that the function wants to keep private, it may be done
    # on premise
    fhe_function = compile_function(function, min_weight, max_weight, num_weights)

    # 1 - This is the key generation, done by the client on its trusted
    # device, once for all
    keys, public_keys = user_generates_its_key(fhe_function)

    # Then, we can use the compiled function and keys for several inputs
    for i in range(2):

        print(f"\nRunning {i}-th test")

        # 2 - Pick inputs on the user device and encrypt them
        #
        # Remark that clear_result is just for comparison between clear and FHE
        # executions, and would not appear in a production kind-of system
        encrypted_weights, clear_result = user_picks_input_and_encrypts(
            function, function_string, keys, min_weight, max_weight, num_weights
        )

        # 3 - This is the FHE execution, done on the untrusted server
        encrypted_result = running_fhe_computation_on_untrusted_server(
            fhe_function, function_string, public_keys, encrypted_weights
        )

        # 4 - This is decryption, done by the client on its trusted device, for
        # each new output. Remark that this function uses keys (ie, not only
        # secret_keys) because it also needs public information
        fhe_result = user_decrypts(keys, encrypted_result)

        # 5 - Finally, for the check and demo, comparing the results. Remark that
        # in a real product, once it is known that FHE results are precise
        # enough
        diff = numpy.abs(fhe_result - clear_result)
        ratio = diff / numpy.max(clear_result)

        print(
            f"\n    Difference between computation in clear and in FHE (expected to be as small as possible): {diff}"
        )
        print(f"    Ratio of difference (expected to be as small as possible): {100 * ratio:.2f} %")


if __name__ == "__main__":
    main()
