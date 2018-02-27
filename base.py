import sys

RANDOM_STATE = 42
PROCESSORS_COUNT = 4
GENERATOR_FOLDER = None
SAVE_RESULTS_AFTER = 10
MAX_EVALS = 500

def forward_out(filename):
    def decorate(func):
        def result(*args, **kwargs):
            stdout = sys.stdout
            stderr = sys.stderr
            try:
                with open(filename, 'a') as out:
                    sys.stdout = sys.stderr = out
                    ans = func(*args, **kwargs)
            finally:
                sys.stdout = stdout
                sys.stderr = stderr
            return ans
        return result
    return decorate



def format_experiment_name(name):
    return name.replace(':', '').replace('/', '')

def get_experiment_name_for_drug(experiment_name, drug):
    return format_experiment_name(
        "{}({})".format(experiment_name, drug),
    )



def get_full_name(model_name, local_name):
    return "{}__{}".format(model_name, local_name)


