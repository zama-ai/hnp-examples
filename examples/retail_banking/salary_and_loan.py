import hnumpy as hnp
from hnumpy.config import CompilationConfig
import numpy


def pay_salaries(accounts_kilo: numpy.ndarray, monthly_salaries_kilo: numpy.ndarray):
    return accounts_kilo + monthly_salaries_kilo


def remove_expanses(accounts_kilo: numpy.ndarray, monthly_expanses_kilo: numpy.ndarray):
    return accounts_kilo - monthly_expanses_kilo


def number_of_positive_values(array_of_values: numpy.ndarray):
    intermediate = numpy.maximum(array_of_values, 0)
    return numpy.sum(numpy.round(intermediate / array_of_values))


def banking_fn(
    accounts_kilo: numpy.ndarray,
    monthly_salaries_kilo: numpy.ndarray,
    monthly_expanses_kilo: numpy.ndarray,
):
    for __ in range(12):
        accounts_kilo = pay_salaries(accounts_kilo, monthly_salaries_kilo)
        accounts_kilo = remove_expanses(accounts_kilo, monthly_expanses_kilo)

    return number_of_positive_values(accounts_kilo)


def main():
    num_accounts = 100
    array_shape = (num_accounts,)
    accounts_kilo = 100.0 * numpy.random.random(array_shape).astype(numpy.float32)
    monthly_salaries_kilo = 8.0 * numpy.random.random(array_shape).astype(numpy.float32)
    monthly_expanses_kilo = 2.0 * numpy.random.random(array_shape).astype(numpy.float32)

    banking_results = banking_fn(accounts_kilo.copy(), monthly_salaries_kilo, monthly_expanses_kilo)

    print(f"Clear: number of accounts with positive balance : {banking_results}")

    config = CompilationConfig(parameter_optimizer="genetic")
    # "handselected"
    # "genetic"

    fhe_banking = hnp.compile_fhe(
        banking_fn,
        {
            "accounts_kilo": hnp.encrypted_ndarray(
                bounds=(accounts_kilo.min(), accounts_kilo.max()),
                shape=accounts_kilo.shape,
            ),
            "monthly_salaries_kilo": hnp.encrypted_ndarray(
                bounds=(monthly_salaries_kilo.min(), monthly_salaries_kilo.max()),
                shape=monthly_salaries_kilo.shape,
            ),
            "monthly_expanses_kilo": hnp.encrypted_ndarray(
                bounds=(monthly_expanses_kilo.min(), monthly_expanses_kilo.max()),
                shape=monthly_expanses_kilo.shape,
            ),
        },
        config=config,
    )

    context = fhe_banking.create_context()
    keys = context.keygen()

    fhe_banking_result = fhe_banking.encrypt_and_run(
        keys,
        accounts_kilo,
        monthly_salaries_kilo,
        monthly_expanses_kilo,
    )[0]

    print(f"FHE: number of accounts with positive balance : {fhe_banking_result}")


if __name__ == "__main__":
    main()
